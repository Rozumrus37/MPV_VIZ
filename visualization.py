import pdb, torch
import time
import os
import math
from torchvision import transforms
import cv2
import kornia
import numpy as np
from IPython.display import HTML, clear_output
import IPython.display
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.image as mpimg
from celluloid import Camera 
import numpy
import ipywidgets as widgets
import tkinter as tk
from tkinter import ttk
from ipywidgets import interact, interactive, fixed, interact_manual
import mpl_interactions.ipyplot as iplt
import time


FULL_ROTATION = 360
SIZE_IMG = 150


def new_func():
  print("yes")
  
def play_with_angle(patch: torch.tensor, orientation_estimation):
  """
  Interactive visualization(with the slider) of working of your orientation_estimation function.

  Args:
    patch: (torch.Tensor) 
    orientation_estimation: estimator function

  Returns:
    nothing, but as side affect patches are shown  
  """

  img = kornia.tensor_to_image(patch)
  fig, ax = plt.subplots(1, 2)
  ax1 = ax[0].imshow(img)
  ax2 = ax[1].imshow(img)
  #plt.close()
  slider = widgets.FloatSlider(value=0, min=0, max=360, step=1, description="Angle:")
  widgets.interact(img_viz, patch=fixed(patch), orientation_estimation=fixed(orientation_estimation), fig=fixed(fig), ax1=fixed(ax1), ax2=fixed(ax2), alfa=slider)


def get_angle(x: torch.Tensor, num_angular_bins: int = 36):
  """
    Function, which estimates the dominant gradient orientation of the given patches, in radians.
    Zero angle points towards right.
    
    Args:
        x: (torch.Tensor) shape (B, 1, PS, PS)
        num_angular_bins: int, default is 36
    
    Returns:
        angles: (torch.Tensor) in radians shape [Bx1]
    """
  estimate = kornia.feature.orientation.PatchDominantGradientOrientation()
  return estimate.forward(x)


# helper function. It is called as a parametr of widgets.interact()
def img_viz(patch: torch.tensor, orientation_estimation, fig, ax1, ax2, alfa=0):
  angle = torch.tensor([np.float32(alfa)])
  patch_rotated = kornia.geometry.transform.rotate(patch, angle)

  estimated_angle = orientation_estimation(patch_rotated).item()

  estim_angle_to_tensor = torch.tensor([(-1)*math.degrees(estimated_angle)])
  patch_out = kornia.geometry.transform.rotate(patch_rotated, estim_angle_to_tensor)

  img1 = kornia.tensor_to_image(patch_rotated)
  img2 = kornia.tensor_to_image(patch_out)

  ax1.set_data(img1)
  ax2.set_data(img2)
  display(fig)
  plt.close()


 
