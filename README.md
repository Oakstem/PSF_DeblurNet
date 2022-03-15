# A Deep Network for Optical Flow Estimation from a Single Blurred Image
 
 
1. Evaluation - In order to run our model in evaluation mode you can run the following code:
run main_train.py --batch_size 3 --pretrained "model_best.pth.tar" --limit 0.06 --dataset "chairs" --evaluate

2. Training - In order to run the training you need to specifiy the location of the dataset and it's name, currently only FlyingChairs / Monkaa datasets are
supported. The dataset should include the pre-processed blurred images and a GT Optical flow images (.pfm / .flo formats supported).
[Monkaa dataset](https://drive.google.com/drive/folders/1PxS7c6BxxjSBKy1kDlmQPcn1ApcZxnkg?usp=sharing)
[FlyingChairs2 dataset](https://drive.google.com/drive/folders/1sMwxn-APwvDv7hQG8vdnRrqkuQ-Do_m0?usp=sharing)

run main_train.py --batch_size 3 --limit 0.06 --dataset "chairs"

3. Pre-trained model - 
[Pre-trained model weights](https://drive.google.com/file/d/11EmloYzKHlgYOdHgMTYz2S_8XD_-E9Qv/view?usp=sharing)
Save the file in FlowNetPytorch/checkpoints/

4. Network architecture in png / onnx formats - [PNG](https://drive.google.com/file/d/1_yuQtzYoVGszKokNGGCZY6S0kX78XSzb/view?usp=sharing), ([Onnx](https://drive.google.com/file/d/1UAXBTNlk4gPK302LdUxRbTXifLlGbce-/view?usp=sharing))

