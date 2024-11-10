[24-SPRING] Machine Learning course final project

*noisy image sequence to text training*

[PROBLEM DEFINITION]
1) dataset has not correct order of image sequence (there is accurately a one swap between adjacent image elements)
2) the label of image sequence is a text sequence which has a correct order
3) the main purpose is to train seq2seq model to infer correct sequence of text even if input image sequence has wrong order


   
[APPROACH]  
process had been improved through 3 steps.  

1) used normal RNN combined with CNN encoder
2) added attention layer on the decoder to utilize image embedding vector, which is on the encoder layer
3) tried transformer based sequence model (but showed lower performance since it needs much more train data to generalize problem, related to it's weak inductive bias)

*) can see description through "./method_description.pdf"
