using Shared;
using TorchSharp;
using TorchSharp.Modules;
using Tensor = TorchSharp.torch.Tensor;
using functional = TorchSharp.torch.nn.functional;
using Device = TorchSharp.torch.Device;
// ReSharper disable SuggestVarOrType_BuiltInTypes
// ReSharper disable SuggestVarOrType_SimpleTypes
// ReSharper disable SuggestVarOrType_Elsewhere
// ReSharper disable InvalidXmlDocComment

// Translated from python code written by Andrej Karpathy https://www.youtube.com/watch?v=kCc8FmEb1nY
// Comments are a mix of my comments, comments from the video, and GPT-4
// Timestamps of video in comments

Device device = torch.cuda.is_available() ? torch.CPU : torch.CPU; // Change to CUDA if you have good gpu and install CUDA driver in shared csproj by uncommenting
if (device.type == DeviceType.CUDA)
{
    torch.InitializeDeviceType(DeviceType.CUDA);
}

torch.manual_seed(1337);

string text = File.ReadAllText("input.txt");

char[] chars = text.Distinct().OrderBy(c => c).ToArray();
var vocabSize = chars.Length;

Console.WriteLine($"Vocab size: {vocabSize}");
Console.WriteLine("Vocab: " + string.Join("", chars));

TokenEncoder encoder = new TokenEncoder(chars);

List<short> encoded = encoder.Encode(text);

Tensor data = torch.tensor(encoded, torch.ScalarType.Int64);

long numberToTrain = (long)(data.shape[0] * 0.9);
long numberToTest = data.shape[0] - numberToTrain;

Tensor trainData = data[..(int)numberToTrain];
Tensor testData = data[(int)numberToTrain..];

Console.WriteLine(numberToTrain);
Console.WriteLine(numberToTest);

DataSampler dataSampler = new DataSampler(trainData, testData);
BigramLanguageModel model = new BigramLanguageModel("My_Language_Model", vocabSize).to(device);

// Timestamp: 35:15
AdamW optimizer = torch.optim.AdamW(model.parameters(), lr: Settings.LearningRate);

for (int i = 0; i < Settings.MaxIterations; i++)
{
    if (i % Settings.EvalInterval == 0)
    {
        float[] losses = EstimateLoss(model, dataSampler, device);
        Console.WriteLine($"step {i}: train loss {losses[0]:F4}, val loss {losses[1]:F4}");
    }

    (Tensor inputs, Tensor targets) = dataSampler.RandomSamples(DataType.Train, Settings.BatchSize, Settings.BlockSize, device);

    (Tensor logits, Tensor? loss) = model.Forward(inputs, targets);
    optimizer.zero_grad();

    // Timestamp: 35:45
    loss?.backward();
    optimizer.step();
}
// Timestamp: 32:15
Tensor context = torch.zeros(new long[] { 1, 1 }, dtype: torch.ScalarType.Int64, device: device);
foreach (var token in model.Generate(context, maxNewTokens: 500))
{
    Console.Write(encoder.Decode(token));
}
Console.WriteLine("\n\n--Complete--");

return;

// Timestamp: 40:00
static float[] EstimateLoss(BigramLanguageModel model, DataSampler dataSampler, Device device)
{
    using var noGrad = torch.no_grad();
    var dataTypes = Enum.GetValues<DataType>();
    float[] results = new float[dataTypes.Length];
    model.eval();
    foreach (var dataType in dataTypes)
    {
        var losses = torch.zeros(Settings.EvalIterations);
        for (int k = 0; k < Settings.EvalIterations - 1; k++)
        {
            var (inputs, targets) = dataSampler.RandomSamples(dataType, Settings.BatchSize, Settings.BlockSize, device);
            var (logits, loss) = model.Forward(inputs, targets);
            losses[k] = loss!.item<float>();
        }
        results[(int)dataType] = losses.mean().item<float>();
    }
    model.train();
    return results;
}

public static class Settings
{
    public const int BatchSize = 32;
    public const int BlockSize = 8;
    public const int MaxIterations = 3000;
    public const int EvalInterval = 300;
    public const double LearningRate = 1e-2;
    public const int EvalIterations = 200;
}

public sealed class BigramLanguageModel : torch.nn.Module
{
    private readonly Embedding _embedding;

    public BigramLanguageModel(string name, long vocabSize) : base(name)
    {
        // Embedding explanation - Timestamp: 23:00
        _embedding = torch.nn.Embedding(vocabSize, vocabSize);
        register_module("token_embedding_table", _embedding);
    }

    public (Tensor logits, Tensor? loss) Forward(Tensor idx, Tensor? targets = null)
    {
        Tensor logits = _embedding.forward(idx); // (B,T,C)
        Tensor? loss = null;

        if (targets is not null)
        {
            var (B, T, C) = (logits.size(0), logits.size(1), logits.size(2));

            // Timestamp: 26:20
            logits = logits.view(B * T, C);
            targets = targets.view(B * T);
            loss = functional.cross_entropy(logits, targets); // cross_entropy expects channel as second dimension
        }

        return (logits, loss);
    }

    // Timestamp: 29:15
    public IEnumerable<short> Generate(Tensor idx, int maxNewTokens)
    {
        for (int i = 0; i < maxNewTokens; i++)
        {
            // get predictions
            (Tensor logits, _) = Forward(idx);
            // focus only on the last time step
            logits = logits.select(1, -1);
            // apply softmax to get probabilities
            Tensor probs = functional.softmax(logits, -1);
            // get next token from the probabilities
            Tensor idxNext = torch.multinomial(probs, 1);
            // append new token to running sequence of tokens 
            idx = torch.cat(new[] { idx, idxNext }, 1);
            yield return (short)idxNext.item<long>();
        }
    }
}