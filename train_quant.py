from utils import data_setup, engine, save
from architecture import Mobilenet, Resnet, Mobilenet_quant
from argparse import ArgumentParser
import torch
from torchinfo import summary
from torchvision import datasets, transforms
from torch.ao.quantization import QuantStub, DeQuantStub, get_default_qconfig, prepare_qat, convert
if __name__ == "__main__":
    parser = ArgumentParser(description='Train classification')
    parser.add_argument('--work-dir', default='models', help='the dir to save logs and models')
    parser.add_argument("--train-folder", default='data/train', type=str)
    parser.add_argument("--valid-folder", default='data/test', type=str)
    parser.add_argument('--architecture', default='MobileNetv3', help='MobileNetv3, ResNet', type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--img-size", default=112, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    args = parser.parse_args()

    print(f'Training {args.architecture} model with hyper-params:')
    devices = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = args.train_folder
    test_dir = args.valid_folder

    IMG_SIZE = args.img_size
    MEANS = (0.46295794, 0.46194877, 0.4847407)
    STDS = (0.19444681, 0.19439201, 0.19383532)
    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS),])
    
    # Load train dataset 
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                transform=manual_transforms, # use manually created transforms
                                                                                batch_size=args.batch_size)
    # Create the model
    if args.architecture == "MobileNetv3":
        model = Mobilenet.MobileNetV3('small')
    elif args.architecture == "Resnet":
        model = Resnet.ResNet(50)
    elif args.architecture == "Mobilenet_quanti":
        model = Mobilenet_quant.MobileNetV3('small')

    # Fuse the Conv, BN, and ReLU modules
    model.conv.c = torch.ao.quantization.fuse_modules(model.conv, ['c', 'bn', 'act'])
    for block in model.blocks:
        block.block[0].c = torch.ao.quantization.fuse_modules(block.block[0], ['c', 'bn', 'act'])
        block.block[1].c = torch.ao.quantization.fuse_modules(block.block[1], ['c', 'bn', 'act'])
        block.block[3].c = torch.ao.quantization.fuse_modules(block.block[3], ['c', 'bn'])

    # Prepare for quantization-aware training
    model.qconfig = get_default_qconfig('fbgemm')
    prepare_qat(model, inplace=True)

    summary(model=model,
            input_size=(128, 3,  112, 112), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # Setup the loss function and optimizer for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Set the seeds
    engine.set_seeds()

    # Train the model and save the training results to a dictionary
    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=args.epochs,
                        device=devices)
    save.save_model(model=model_fp32_prepared,
                target_dir=args.work_dir,
                model_name=args.architecture + "_before.pth")
   
    # Convert to a quantized model
    model.eval()
    model = convert(model, inplace=True)
    print(f'Check statistics of the various layers')
    print(model_int8)
    # Print the weights matrix of the model before quantization
    # print('Weights before quantization')
    # print(torch.int_repr(model_quantized.linear1.weight()))
    # Save the model with help from utils.py
    save.save_model(model=model,
                    target_dir=args.work_dir,
                    model_name=args.architecture + "_after.pth")