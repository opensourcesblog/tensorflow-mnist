import predict_interface
import numpy as np
import sys

"""
How to use the predict_interface -interface
"""

retVals = []
image = sys.argv[1]
train = False if len(sys.argv) == 2 else sys.argv[2]

ret_vals = predict_interface.pred_from_img(image, train)

print(" ")
print("----------")
	
for i in range(0, len(ret_vals)):
	print("Prediction: " + str(ret_vals[i].get_prediction()) + ", Probability: " + str(ret_vals[i].get_probability()))
	print("Location in image: x(" + str(ret_vals[i].get_location()[1]) + "), y(" + str(ret_vals[i].get_location()[0]) + ")")
	print("Top left corner (shifted image): x(" + str(ret_vals[i].get_top_left()[1]) + "), y(" + str(ret_vals[i].get_top_left()[0]) + ")")
	print("Bottom right corner (shifted image): x(" + str(ret_vals[i].get_bottom_right()[1]) + "), y(" + str(ret_vals[i].get_bottom_right()[0]) + ")")
	print("Actual width and height (cropped image): w(" + str(ret_vals[i].get_actual_w_h()[1]) + "), h(" + str(ret_vals[i].get_actual_w_h()[0]) + ")")
	print(" ")
	print("----------")

print("A modified image with the predictions: pro-img/IMAGE_NAME_digitized_image.png")