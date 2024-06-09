from utils import data_setup, engine, save
from architecture import Mobilenet, Resnet, Mobilenet_quant
from argparse import ArgumentParser
import torch
from torchvision import datasets, transforms
from torch.quantization import QuantStub, DeQuantStub, default_qconfig, prepare_qat, convert

def main():
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

    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.46295794, 0.46194877, 0.4847407), (0.19444681, 0.19439201, 0.19383532))
    ])

    # Load train dataset
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=args.train_folder,
        test_dir=args.valid_folder,
        transform=manual_transforms,
        batch_size=args.batch_size
    )

    # Create the model
    if args.architecture == "MobileNetv3":
        model = Mobilenet.MobileNetV3('small')
    elif args.architecture == "Resnet":
        model = Resnet.ResNet(50)
    elif args.architecture == "Mobilenet_quanti":
        model = Mobilenet_quant.MobileNetV3('small')

    # Move the model to the selected device
    model.to(devices)

    # Specify quantization configuration for QAT
    quant_config = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare the model for quantization
    model.qconfig = quant_config
    q_model = prepare_qat(model, inplace=False)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(q_model.parameters(), lr=args.lr, momentum=0.9)

    # Set the seeds
    engine.set_seeds()

    # Train the model and save the training results to a dictionary
    results = engine.train(
        model=q_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        work_dir=args.work_dir,
        architecture=args.architecture,
        device=devices
    )

    # Convert the model to quantized form after training
    quantized_model = convert(q_model.eval(), inplace=False)

    # Save the quantized model
    save.save_model(quantized_model, args.work_dir, model_name=f"{args.architecture}_quantized.pth")

if __name__ == "__main__":
    main()