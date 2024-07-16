# Pytorch-cifar100

Practice on Canjie dataset using pytorch, I modified SEResNet34 to run the dataset.

## Requirements
```
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 Pillow matplotlib -c pytorch -c nvidia
$ pip install -r requirements.txt
```

## Usage

### 1. Download Dataset
- Download the canjie dataset from [here](https://drive.google.com/file/d/1AIi8286VY4QGJ8i2SY1sIQc7OeUdJqwX/view) and extract to `./data` directory
- Download the pre-trained model and put into `./checkpoint` from [here](https://drive.google.com/drive/folders/18Wpo_XCJRhWWzdR5Q0_rCOToEJ5M17v0?usp=sharing) and extract to `./data` directory

### 2. Training model

#### For Task 1
You can use `train.py` as following command to train the SEResNet34

```bash
python train.py -net seresnet34 -gpu -data_dir="./data"
```
#### For Task 2

```bash
python train.py -net custom_seresnet34 -gpu -data_dir="./data"
```

- The `./results` directory already contains logging information of training and testing, for example `train_before.txt` is logging data when training default seresnet34 without any modifications, `test_success5.txt` contains testing results with custom_seresnet34 model which modified at the fixth time.

### 3. Use pre-train model

Pre-train model is included in `./checkpoint` directory

#### For Task 1
You can use `train.py` as following command to train the SEResNet34

```bash
python train.py -net seresnet34 -gpu -data_dir="./data" -weights ./checkpoint/seresnet34/training/seresnet34-6-best.pth
```
#### For Task 2

```bash
python train.py -net custom_seresnet34 -gpu -data_dir="./data" -weights ./checkpoint/custom_seresnet34/6th_attempt/custom_seresnet34-6-best.pth
```

### 4. Verify test results

To verify test result, you can run below command to eval the model in testset which default should be located in `./data/etl_952_singlechar_size_64/952_test`

```bash
python test.py -net custom_seresnet34 -gpu -data_dir="./data" -weights ./checkpoint/custom_seresnet34/6th_attempt/custom_seresnet34-6-best.pth
```

### 5. Task 3 - detecting canjie character

Question3 requires passing image to modified model to inference encoding character of that image.

```bash
python question3.py --image ./assets/test_image/i_hiragana.png –txt_file ./data/etl_952_singlechar_size_64/952_labels.txt --model ./checkpoint/custom_seresnet34/6th_attempt/custom_seresnet34-6-best.pth
```

### More details

All the reports were published in [my cinnamon report](https://docs.google.com/document/d/1XNTwOLNeldHKFZgyRYY3yHOJvxSA9ZNEkdk9hNANldQ/edit?usp=sharing), please take a look to see all the performance records.

