"""Retrainer that uses the most recent data points to retrain the model."""
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from drift_detection.baseline_models.temporal.pytorch.metrics import (
    print_metrics_binary,
)
from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import get_data
from drift_detection.drift_detector.detector import Detector
from drift_detection.gemini.utils import process
import numpy as np

TEMPORAL_MODELS = ["rnn","gru","lstm"]

class PeriodicRetrainer:
    """Retrainer that uses the most recent data points to retrain the model.

    Attributes
    ----------
    shift_detector : Detector
        Detector object for detecting data shift.
    optimizer : Optimizer
        Optimizer object for training the model.
    model : torch.nn.Module
        Model to be trained.
    model_name : str
        Name of the model.
    retrain_model_path : str
        Path to save the retrained model.
    verbose : int
        Whether to print out the training progress.

    Methods
    -------
    retrain(X_s, X_t, **kwargs)
        Retrains the model on the target data.

    """

    def __init__(
        self,
        shift_detector: Detector,
        optimizer: Optimizer,
        model=None,
        model_name: str = None,
        retrain_model_path: str = None
    ):

        self.shift_detector = shift_detector
        self.optimizer = optimizer
        self.model = model
        self.model_name = model_name
        self.retrain_model_path = retrain_model_path

    def retrain(
        self,
        data_streams: dict = None,
        freq: int = 30,
        retrain_window: int = 30,
        sample: int = 1000, 
        stat_window: int = 30, 
        lookup_window: int = 0, 
        stride: int = 1, 
        batch_size: int = 64,
        n_epochs: int = 1,
        aggregation_type = "time",
        correct_only: int = 0,
        positive_only: int = 0,
        verbose: int = 1,
        **kwargs
    ):
        
        """Retrain the model on the target data.

        Parameters
        ----------
        data_streams : dict
            Dictionary of data streams.
        retrain_window : int
            Number of days to retrain the model.
        sample : int
            Number of samples to use for retraining.
        stat_window : int
            Number of days to compute the statistics.
        lookup_window : int
            Number of days to look ahead for the shift.
        stride : int
            Stride for the rolling window.
        freq : int
            Number of days to retrain at.
        batch_size : int
            Batch size for training.
        n_epochs : int
            Number of epochs to train the model.
        aggregation_type: str
            How to aggregate data (e.g. time, mean)
        correct_only: bool
            Whether to use all samples for retraining or only those predicted correctly by the model.

        Returns
        -------
        results : dict
            Dictionary of retraining drift and performance results.

        """
        
        rolling_metrics = []
        run_length = stat_window 
        i = 0
        
        num_timesteps = data_streams['X'][0].index.get_level_values(1).nunique()
        n_features = data_streams['X'][0].shape[1]
        pbar_total=len(data_streams['X'])-stat_window-lookup_window+1
        pbar = tqdm(total = pbar_total, miniters = int(pbar_total/100))
        
        print("Calibrating drift detector...")
        
        while i+stat_window+lookup_window  < len(data_streams['X']):
            
            if (i % 50 == 0):
                pbar.update(50)
            
            if (i % freq == 0) and (i != 0):

                if i > stat_window*2+7:
                    
                    X_update_streams = pd.concat(
                        data_streams['X'][max(int(i)-run_length,0):int(i)]
                    )
                    X_update_streams = X_update_streams[~X_update_streams.index.duplicated(keep='first')]
                    ind = X_update_streams.index.get_level_values(0).unique()
                    encounter_ids = np.repeat(ind, num_timesteps)            
                    X_update = process(X_update_streams, aggregation_type, num_timesteps)   

                    y_update = pd.concat(data_streams['y'][max(int(i)-run_length,0):int(i)])
                    y_update.index = ind
                    y_update = y_update[~y_update.index.duplicated(keep='first')]
                    y_update_final = y_update.to_numpy()[:, :, np.newaxis]

                    assert not np.any(np.isnan(X_update))
                    assert not np.any(np.isnan(y_update_final))

                    if self.model_name in TEMPORAL_MODELS:

                        ## Remove all incorrectly predicted labels for retraining
                        if correct_only == 1 or positive_only ==1:
                            update_loader = get_data(X_update, y_update_final).to_loader(batch_size,shuffle=True)
                            y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                                update_loader
                            )
                            
                            
                            if correct_only:
                                y_pred_values = y_pred_values[y_pred_labels == y_test_labels]
                                encounter_ids = encounter_ids[y_pred_labels == y_test_labels]
                                y_test_labels = y_pred_labels = y_pred_labels[y_pred_labels == y_test_labels]
                            elif positive_only:
                                y_pred_values = y_pred_values[y_pred_labels == 1]
                                encounter_ids = encounter_ids[y_pred_labels == 1]
                                y_test_labels = y_pred_labels = y_pred_labels[y_pred_labels == 1]                   

                            assert len(encounter_ids) == len(y_test_labels) 

                            X_update_streams = X_update_streams.loc[X_update_streams.index.get_level_values(0).isin(encounter_ids)]
                            X_update = process(X_update_streams, aggregation_type, num_timesteps)
                            y_update = y_update.loc[y_update.index.isin(encounter_ids)]
                            y_update_final = y_update.to_numpy()[:, :, np.newaxis]

                        X_val, X_test, y_val, y_test = train_test_split(X_update, y_update_final, test_size=0.20, random_state=42)
                        val_loader =  get_data(X_val, y_val).to_loader(batch_size, shuffle=True)
                        test_loader =  get_data(X_test, y_test).to_loader(batch_size, shuffle=True)

                        if self.retrain_model_path is None:
                            self.retrain_model_path="_".join(["periodic",str(retrain_window),str(stat_window),str(n_epochs),str(sample),str(freq),"retrain.model"])
                        
                        if verbose == 1:
                            print("Retrain ",self.model_name," on: ",data_streams['timestamps'][max(int(i)-run_length,0)],"-",data_streams['timestamps'][int(i)])
                    
                        ## Update the model
                        try:
                            self.optimizer.train(
                                val_loader,
                                test_loader,
                                n_epochs=n_epochs,
                                model_path=self.retrain_model_path,
                            )
                            self.model.load_state_dict(torch.load(self.retrain_model_path))
                            self.shift_detector.reductor.model = self.optimizer.model = self.model
                            self.shift_detector.reductor.model_path = self.retrain_model_path
                        
                        except AssertionError:
                            pass

                    elif self.model_name == "gbt":
                        self.model = self.model.fit(X_retrain, y_retrain, xgb_model=self.model.get_booster())

                    else:
                        print("Invalid Model Name")

            X_next = pd.concat(data_streams['X'][max(int(i)+lookup_window,0):int(i)+stat_window+lookup_window])
            X_next = X_next[~X_next.index.duplicated(keep='first')]
            next_ind = X_next.index.get_level_values(0).unique()
            X_next = process(X_next, aggregation_type, num_timesteps)

            y_next = pd.concat(data_streams['y'][max(int(i)+lookup_window,0):int(i)+stat_window+lookup_window])
            y_next.index = next_ind
            y_next = y_next[~y_next.index.duplicated(keep='first')].to_numpy()[:, :, np.newaxis]

            ## Check if there are patient encounters in the next timestep
            if X_next.shape[0]<=2:
                break

            ## Check Performance 
            test_dataset = get_data(X_next, y_next)
            test_loader = test_dataset.to_loader(batch_size=1, shuffle=False)
            y_test_labels, y_pred_values, y_pred_labels = self.optimizer.evaluate(
                test_loader,
            )
            assert y_test_labels.shape == y_pred_labels.shape == y_pred_values.shape
            y_pred_values = y_pred_values[y_test_labels != -1]
            y_pred_labels = y_pred_labels[y_test_labels != -1]
            y_test_labels = y_test_labels[y_test_labels != -1]  

            performance_metrics = print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=0)

            ## Detect Distribution Shift 
            drift_metrics = self.shift_detector.detect_shift(
                X_next,
                sample,
                **kwargs
            )
            p_val = drift_metrics['p_val']
            metrics = {
                **drift_metrics, 
                **performance_metrics
            }
            rolling_metrics.append(metrics)
            
            i += stride 
            
            if i % freq != 0:
                run_length += stride 
            else:
                run_length= retrain_window        
                if verbose == 1 and i > stat_window*2+7:
                    print(
                        "Triggered at ",data_streams['timestamps'][i+lookup_window],"-",data_streams['timestamps'][i+stat_window+lookup_window],"\tP-Value: ",drift_metrics['p_val'])

        pbar.close()

        rolling_metrics = {
            k: [d.get(k) for d in rolling_metrics] for k in set().union(*rolling_metrics)
        }
        
        return rolling_metrics
