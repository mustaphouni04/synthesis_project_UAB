import wandb
import torch

def test(model, test_loader, device="cuda", save=True):
    # Run the model on some test examples
    with torch.no_grad():
        mse_total = 0
        for inputs in test_loader:  # Directly iterate over inputs
            inputs = inputs.to(device)
            outputs = model(inputs)
            mse = torch.mean((outputs - inputs)**2, dim=(1, 2))  # Compute MSE across batches
            mse_total += mse.sum().item()

        avg_mse = mse_total / len(test_loader.dataset)
        print(f"Average MSE of the model on the test set: {avg_mse}")

        wandb.log({"test_avg_mse": avg_mse})

    if save:
        # Save the model in the exchangeable ONNX format
        # Note: Adjusting input and output names for autoencoder
        dummy_input = torch.randn(1, 20, 25)
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['reconstructed_output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'reconstructed_output': {0: 'batch_size'}})
        wandb.save("model.onnx")