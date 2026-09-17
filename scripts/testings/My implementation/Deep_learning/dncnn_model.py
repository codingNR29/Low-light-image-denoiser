import torch
import torch.nn as nn


#DnCNN model
class DnCNN(nn.Module):

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        num_features=64,
        num_layers=17
    ):

        #Initializing parent PyTorch module
        super().__init__()


        #List for storing all network layers
        layers = []


        #First convolution layer
        #Input RGB image has 3 channels
        #Output has 64 feature maps
        layers.append(

            nn.Conv2d(

                in_channels=input_channels,

                out_channels=num_features,

                kernel_size=3,

                padding=1,

                bias=True

            )

        )


        #ReLU activation
        layers.append(

            nn.ReLU(
                inplace=True
            )

        )


        #Middle convolution layers
        #Each middle layer keeps 64 feature maps
        for _ in range(
            num_layers - 2
        ):

            #Convolution layer
            layers.append(

                nn.Conv2d(

                    in_channels=num_features,

                    out_channels=num_features,

                    kernel_size=3,

                    padding=1,

                    bias=False

                )

            )


            #Batch normalization
            layers.append(

                nn.BatchNorm2d(
                    num_features
                )

            )


            #ReLU activation
            layers.append(

                nn.ReLU(
                    inplace=True
                )

            )


        #Final convolution layer
        #Converts 64 feature maps back to 3 RGB residual channels
        layers.append(

            nn.Conv2d(

                in_channels=num_features,

                out_channels=output_channels,

                kernel_size=3,

                padding=1,

                bias=True

            )

        )


        #Combining all layers into one sequential model
        self.network = nn.Sequential(
            *layers
        )


    #Forward pass
    def forward(
        self,
        noisy
    ):

        #Predicting the noise residual
        predicted_residual = self.network(
            noisy
        )


        return predicted_residual



#Testing the model
if __name__ == "__main__":

    #Selecting GPU if available
    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        "Using device:",
        device
    )


    #Creating DnCNN model
    model = DnCNN(

        input_channels=3,

        output_channels=3,

        num_features=64,

        num_layers=17

    )


    #Moving model to GPU or CPU
    model = model.to(
        device
    )


    #Creating one fake batch
    #Batch size = 16
    #RGB channels = 3
    #Patch size = 64x64
    dummy_input = torch.randn(

        16,
        3,
        64,
        64,

        device=device

    )


    #Running fake batch through model
    predicted_residual = model(
        dummy_input
    )


    #Reconstructing clean image
    predicted_clean = (

        dummy_input
        -
        predicted_residual

    )


    #Printing shapes
    print(
        "Input shape:",
        dummy_input.shape
    )


    print(
        "Predicted residual shape:",
        predicted_residual.shape
    )


    print(
        "Predicted clean shape:",
        predicted_clean.shape
    )


    #Counting trainable parameters
    total_parameters = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )


    print(
        "Trainable parameters:",
        total_parameters
    )


    #Checking GPU memory
    if torch.cuda.is_available():

        allocated_memory = (

            torch.cuda.memory_allocated()

            / 1024**2

        )


        reserved_memory = (

            torch.cuda.memory_reserved()

            / 1024**2

        )


        print(
            "GPU allocated memory:",
            f"{allocated_memory:.2f} MB"
        )


        print(
            "GPU reserved memory:",
            f"{reserved_memory:.2f} MB"
        )