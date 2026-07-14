import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

import torch

class DataPreparation():
       
  def __init__(self, edf_path, timelog_path, ibi_path=None, signal250hz_path=None,
               fs=250):
     
    """Class for preparing the dataset.
    Initialize the DataPreparation class with paths to data files and sampling frequency.
    
    Args:
        edf_path (str): Path to the folder containing EDF files.
        timelog_path (str): Path to the folder containing timelog files.
        ibi_path (str, optional): Path to the folder containing IBI files.
        signal250hz_path (str, optional): Path to the folder containing 250Hz signal files.
        fs (int): Sampling frequency.
    
    results:
        Creates a DataFrame containing the common subject IDs and their corresponding file names.
    """
    
    
    self.edf_path = edf_path
    self.timelog_path = timelog_path
    self.ibi_path = ibi_path
    self.signal250hz_path = signal250hz_path
    self.fs = fs

    edf_folder= Path(edf_path)
    txt_folder= Path(timelog_path)
    ibi_folder= Path(ibi_path) if ibi_path else None
    signal250hz_folder= Path(signal250hz_path) if signal250hz_path else None

    edf_files = list(edf_folder.glob("*.edf"))
    txt_files = list(txt_folder.glob("*.txt"))
    ibi_files = list(ibi_folder.glob("*.txt")) if ibi_folder else []
    signal250hz_files = list(signal250hz_folder.glob("*.txt")) if signal250hz_folder else []

    ibi_df = [file.stem[:8] for file in ibi_files]
    edf_df = [file.stem[:8] for file in edf_files]
    txt_df = [file.stem[:8] for file in txt_files]
    signal250hz_df = [file.stem[:8] for file in signal250hz_files]
    
    common_ids = sorted( set(ibi_df) & set(edf_df) & set(txt_df) & set(signal250hz_df))
    
    data_files= pd.DataFrame(columns=["sbj_id","edf_name","Timelog_name", "IBI_name", "Signal250Hz_name"])
    for sbj_id in common_ids:
        edf_name = next((file.stem for file in edf_files if file.stem.startswith(sbj_id)), None)
        txt_name = next((file.stem for file in txt_files if file.stem.startswith(sbj_id)), None)
        ibi_name = next((file.stem for file in ibi_files if file.stem.startswith(sbj_id)), None)
        signal250hz_name = next((file.stem for file in signal250hz_files if file.stem.startswith(sbj_id)), None)
        temp= pd.DataFrame([[sbj_id, edf_name, txt_name, ibi_name, signal250hz_name]], columns=["sbj_id","edf_name","Timelog_name", "IBI_name", "Signal250Hz_name"])
        data_files= pd.concat([data_files,temp])


   
    data_files.reset_index(drop=True, inplace=True)
    data_files.to_csv(r"dataset\files_list.csv", index=False)
    self.data_files=data_files
    self.subjects= pd.unique(data_files["sbj_id"])      

  def create_dataset(self):
    """Create a dataset based on the data_files DataFrame
    This is a placeholder for the actual dataset creation logic.
    
    returns:
        input (np.ndarray): The input data array.
        target (np.ndarray): The target data array."""
    
    
    X_list= []
    Y_list= []
    real_target= []
    files_list = pd.read_csv(r"dataset\files_list.csv")

    for sbj_id in self.subjects:
        
        
        ibi = files_list.loc[files_list['sbj_id'] == sbj_id]['IBI_name'].values[0]+'.txt'
        ibis_df = pd.read_csv("E:\Bradshaw_HRfiles\IBI" + "\\" + ibi, sep=r"\s+", header=None)
        ibis_df.columns = ["chron_time", "time_start_1", "time_start_2", "ibi_ms"]

        timelog = files_list.loc[files_list['sbj_id'] == sbj_id]['Timelog_name'].values[0]+'.txt'
        timelog_df= pd.read_csv("E:\Bradshaw_HRfiles\Timelog"+ "\\"+ timelog)
        timelog_df.columns= ["Visit Date","Segment","Condition","Start","End"]

        sig250= files_list.loc[files_list['sbj_id'] == sbj_id]['Signal250Hz_name'].values[0]+'.txt'
        signal250hz= pd.read_csv(r"E:\Bradshaw_HRfiles\250Hz"+ "\\"+ sig250, sep=r"\s+", header=None)

        baseline = timelog_df[timelog_df["Condition"] == "Baseline"]
        if not baseline.empty:
            start_sec,end_sec= timelog_df.iloc[0][["Start","End"]].astype(float)*60 
            #rpeaks=(ibis_df.loc[(ibis_df.iloc[:,1]> start_sec) & (ibis_df.iloc[:,1] < end_sec)]['time_start_1'] * 250).astype(int)  
            rpeaks= (ibis_df.loc[(ibis_df.iloc[:,1]> start_sec) & (ibis_df.iloc[:,1] < end_sec)]["time_start_1"] * 250).astype(int)  

            sig = signal250hz[(signal250hz.iloc[:, 1] > start_sec) & (signal250hz.iloc[:, 1] < end_sec)][signal250hz.columns[3]].values
            startsample= int(start_sec * self.fs)
            endsample= int(end_sec * self.fs)
            sig= signal250hz.iloc[startsample:endsample,3]

            # Calculate the absolute end index for the 'sig' data
            sig_end_abs_idx = endsample  

            # Filter R-peaks that fall within the absolute range of the 'sig' data
            rpeaks_in_window_abs = rpeaks[(rpeaks >= startsample) & (rpeaks <= sig_end_abs_idx)]
            #print(rpeaks_in_window_abs)
            # Convert these absolute indices to relative indices for plotting within the 'sig' array
            rpeaks_relative_to_sig = rpeaks_in_window_abs - startsample
            #print(len(rpeaks_relative_to_sig), len(sig))
            target= self.make_rpeak_target(len(sig),rpeaks_relative_to_sig, self.fs, sigma_ms=20)
            x,y, real_tar = self.create_windows(sig, target, rpeaks_relative_to_sig, self.fs, window_sec=2, stride_sec=1)
            #print(x.shape,y.shape, real_tar.shape)

            X_list.append(x)
            Y_list.append(y)
            real_target.append(real_tar)
            #print(" ")
            # target = np.zeros(len(rpeaks_relative_to_sig), dtype=np.int32)
            # target[:] = rpeaks_relative_to_sig
            # real_target.append(target)

    
    input = np.concatenate(X_list, axis=0)
    target = np.concatenate(Y_list, axis=0)
    
    real_target_flat = []

    for subject_targets in real_target:
        for window_targets in subject_targets:
            window_targets = np.asarray(
                window_targets,
                dtype=np.int64,
            ).reshape(-1)

            real_target_flat.append(window_targets)

    real_target = np.asarray(real_target_flat, dtype=object)

    print("Input shape:", input.shape)
    print("Target shape:", target.shape)
    print("Number of real targets:", len(real_target))
    #real_target = np.concatenate(real_target, axis=0)
  
    return input, target, real_target


  def make_rpeak_target(self,signal_length, rpeaks, fs=250, sigma_ms=20):
        """
        Create Gaussian target peaks centered at R-peak locations.
        
        returns:
            target (np.ndarray): An array of the same length 
            as the input signal, with Gaussian peaks at R-peak locations.
        """

        target = np.zeros(signal_length, dtype=np.float32)
        sigma = max(1, int(round((sigma_ms / 1000) * fs)))
        radius = int(round(4 * sigma))
        rpeaks = np.asarray(rpeaks, dtype=int)
        #print(rpeaks)
        for rp in rpeaks:
            if rp < 0 or rp >= signal_length:
                continue

            left = max(0, rp - radius)
            right = min(signal_length, rp + radius + 1)
            
            idx = np.arange(left, right)
            gaussian = np.exp(-0.5 * ((idx - rp) / sigma) ** 2).astype(np.float32)
            
            target[left:right] = np.maximum(target[left:right], gaussian)
            
        return target

     
  def create_windows(self,ecg, target, rpeaks, fs=250, window_sec=10, stride_sec=5):
      """Create sliding windows from the ECG signal and corresponding targets.
      
      returns:
          X (np.ndarray): Array of ECG windows.
      """
      
      window_size = int(window_sec * fs)
      stride = int(stride_sec * fs)


      X = []
      Y = []
      real_targets = []
      for start in range(0, len(ecg) - window_size + 1, stride):
          end = start + window_size

          
           # Select global R-peaks inside this window.
          mask = (rpeaks >= start) & (rpeaks < end)
          #print(mask)
          global_window_rpeaks = rpeaks[mask]
          #print(global_window_rpeaks)

            # Convert global sample numbers into local window positions.
          local_rpeaks = global_window_rpeaks - start
          x_win = ecg[start:end].astype(np.float32)
          y_win = target[start:end].astype(np.float32)
          
          X.append(x_win)
          Y.append(y_win)
          real_targets.append(local_rpeaks)
          
      #print(local_rpeaks)
      X = np.array(X)
      Y = np.array(Y)
      real_targets = np.array(real_targets, dtype=object)
      
      print(len(X), len(Y), len(real_targets))

      return X, Y, real_targets


dp= DataPreparation(edf_path=r"E:\Bradshaw_HRfiles\EDF", timelog_path=r"E:\Bradshaw_HRfiles\Timelog", ibi_path=r"E:\Bradshaw_HRfiles\IBI", signal250hz_path=r"E:\Bradshaw_HRfiles\250Hz", fs=250)
x,y, real_target= dp.create_dataset()

np.save(r"dataset\input2.npy", x)
np.save(r"dataset\target2.npy", y)
np.save(r"dataset\real_target2.npy", real_target)