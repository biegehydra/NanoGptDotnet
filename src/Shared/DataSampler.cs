using TorchSharp;

namespace Shared;

public class DataSampler
{
    private readonly torch.Tensor _trainData;
    private readonly torch.Tensor _testData;
    public DataSampler(torch.Tensor trainData, torch.Tensor testData)
    {
        _trainData = trainData;
        _testData = testData;
    }

    public (torch.Tensor inputs, torch.Tensor targets) RandomSample(DataType dataType, int batchSize, int blockSize, torch.Device device)
    {
        var data = dataType == DataType.Train ? _trainData : _testData;
        var dimension = new long[] { batchSize };
        torch.Tensor randTensor = torch.randint(0, data.shape[0] - blockSize, dimension);
        List<torch.Tensor> inputs = new List<torch.Tensor>();
        List<torch.Tensor> targets = new List<torch.Tensor>();

        for (int i = 0; i < batchSize; i++)
        {
            var startIdx = (int)randTensor[i].item<long>();
            torch.Tensor inputBlock = data[startIdx..(startIdx + blockSize)];
            torch.Tensor targetBlock = data[(startIdx + 1)..(startIdx + blockSize + 1)];

            inputs.Add(inputBlock);
            targets.Add(targetBlock);
        }

        // Convert lists of Tensors to a single Tensor
        torch.Tensor x = torch.stack(inputs).to(device);
        torch.Tensor y = torch.stack(targets).to(device);
        return (x, y);
    }
}