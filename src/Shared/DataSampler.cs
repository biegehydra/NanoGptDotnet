using TorchSharp;

namespace Shared;

public class DataSampler
{
    private readonly torch.Tensor _trainData;
    private readonly torch.Tensor _testData;

    // Timestamp: 19:00
    public DataSampler(torch.Tensor trainData, torch.Tensor testData)
    {
        _trainData = trainData;
        _testData = testData;
    }

    /// <summary>
    /// Gets random samples from either the train data or test data.
    /// The batch size controls how many samples to take and the
    /// block size controls how many tokens will be in each sample.
    /// Tokens in a block are contiguous but the batches may or may
    /// not be contiguous (MOST likely not). 
    /// </summary>
    public (torch.Tensor inputs, torch.Tensor targets) RandomSamples(DataType dataType, int batchSize, int blockSize, torch.Device device)
    {
        var data = dataType == DataType.Train ? _trainData : _testData;
        var dimension = new long[] { batchSize };
        // Rand tensor will contain the batchSize random numbers representing
        // start indexes for random blocks.
        // data.shape[0] - blockSize makes sure that room is left for the block
        torch.Tensor randTensor = torch.randint(0, data.shape[0] - blockSize, dimension);
        List<torch.Tensor> inputs = new List<torch.Tensor>();
        List<torch.Tensor> targets = new List<torch.Tensor>();

        for (int i = 0; i < batchSize; i++)
        {
            // Get a random start index from the rand tensor
            var startIdx = (int)randTensor[i].item<long>();
            
            // Get a contiguous block of tokens starting at that random index
            torch.Tensor inputBlock = data[startIdx..(startIdx + blockSize)];

            // The target block is just the input block but with the window shifted
            // to the right one: targetBlock[i] = inputBlock[i+1]
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