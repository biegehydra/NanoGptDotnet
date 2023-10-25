using System.Collections;
using TorchSharp;
using TorchSharp.Modules;
using Tensor = TorchSharp.torch.Tensor;
using functional = TorchSharp.torch.nn.functional;
using Device = TorchSharp.torch.Device;

Device device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
torch.InitializeDeviceType(DeviceType.CUDA);

torch.manual_seed(1337);

string text = File.ReadAllText("input.txt");

char[] chars = text.Distinct().OrderBy(c => c).ToArray();
var vocabSize = chars.Length;

Console.WriteLine($"Vocab size: {vocabSize}");
Console.WriteLine("Vocab: " + string.Join("", chars));

Encoding encoding = new Encoding(chars);

List<short> encoded = encoding.Encode(text);

Tensor data = torch.tensor(encoded, torch.ScalarType.Int64);

long numberToTrain = (long) (data.shape[0] * 0.9);
long numberToTest = data.shape[0] - numberToTrain;

Tensor trainData = data[..(int) numberToTrain];
Tensor testData = data[(int) numberToTrain..];

Console.WriteLine(numberToTrain);
Console.WriteLine(numberToTest);

BatchDispatcher batchDispatcher = new BatchDispatcher(trainData, testData);
BigramLanguageModel model = new BigramLanguageModel("My_Language_Model", vocabSize).to(device);
AdamW optimizer = torch.optim.AdamW(model.parameters(), lr: Consts.LearningRate);

for (int i = 0; i < Consts.MaxIterations; i++)
{
    if (i % Consts.EvalInterval == 0)
    {
        float[] losses = EstimateLoss(model, batchDispatcher, device);
        Console.WriteLine($"step {i}: train loss {losses[0]:F4}, val loss {losses[1]:F4}");
    }

    (Tensor inputs, Tensor targets) = batchDispatcher.GetBatch(DataType.Train, device);

    (Tensor logits, Tensor? loss) = model.Forward(inputs, targets);
    optimizer.zero_grad();
    loss?.backward();
    optimizer.step();
}

var context = torch.zeros(new long[] {1, 1}, dtype: torch.ScalarType.Int64, device: device);
var generation = (ArrayList) model.Generate(context, maxNewTokens: 500)[0].tolist();
List<short> results = new List<short>();
foreach (var obj in generation)
{
    var scalar = (Scalar) obj;
    results.Add(scalar.ToInt16());
}
Console.WriteLine(encoding.Decode(results));

static float[] EstimateLoss(BigramLanguageModel model, BatchDispatcher batchDispatcher, Device device)
{
    var dataTypes = Enum.GetValues<DataType>();
    float[] results = new float[dataTypes.Length];
    model.eval();
    foreach (var dataType in dataTypes)
    {
        var losses = torch.zeros(Consts.EvalIterations);
        for (int k = 0; k < Consts.EvalIterations - 1; k++)
        {
            var (inputs, targets) = batchDispatcher.GetBatch(dataType, device);
            var (logits, loss) = model.Forward(inputs, targets);
            losses[k] = loss?.item<float>() ?? 0f;
        }
        results[(int)dataType] = losses.mean().item<float>();
    }
    return results;
}
public enum DataType
{
    Train,
    Test
}

public static class Consts
{
    public const int BatchSize = 32;
    public const int BlockSize = 8;
    public const int MaxIterations = 3000;
    public const int EvalInterval = 300;
    public const double LearningRate = 1e-2;
    public const int EvalIterations = 200;
}

public class BatchDispatcher
{
    private readonly Tensor TrainData;
    private readonly Tensor TestData;
    public BatchDispatcher(Tensor trainData, Tensor testData)
    {
        TrainData = trainData;
        TestData = testData;
    }

    public (Tensor inputs, Tensor targets) GetBatch(DataType dataType, Device device)
    {
        var data = dataType == DataType.Train ? TrainData : TestData;
        var dimension = new long[] { Consts.BatchSize };
        Tensor randTensor = torch.randint(0, data.shape[0] - Consts.BlockSize, dimension);
        List<Tensor> inputs = new List<Tensor>();
        List<Tensor> targets = new List<Tensor>();

        for (int i = 0; i < Consts.BatchSize; i++)
        {
            var startIdx = (int) randTensor[i].item<long>();
            Tensor inputBlock = data[startIdx..(startIdx + Consts.BlockSize)];
            Tensor targetBlock = data[(startIdx + 1)..(startIdx + Consts.BlockSize + 1)];

            inputs.Add(inputBlock);
            targets.Add(targetBlock);
        }

        // Convert lists of Tensors to a single Tensor
        Tensor x = torch.stack(inputs).to(device);
        Tensor y = torch.stack(targets).to(device);
        return (x, y);
    }
}

public sealed class BigramLanguageModel : torch.nn.Module
{
    private Embedding _embedding;

    public BigramLanguageModel(string name, long vocabSize) : base(name)
    {
        _embedding = torch.nn.Embedding(vocabSize, vocabSize);
        register_module("token_embedding_table", _embedding);
    }

    public (Tensor logits, Tensor? loss) Forward(Tensor idx, Tensor targets = null)
    {
        Tensor logits = _embedding.forward(idx); // (B,T,C)
        Tensor? loss = null;

        if (!ReferenceEquals(targets, null))
        {
            var (B, T, C) = (logits.size(0), logits.size(1), logits.size(2));
            logits = logits.view(B * T, C);
            targets = targets.view(B * T);
            loss = functional.cross_entropy(logits, targets);
        }

        return (logits, loss);
    }

    public Tensor Generate(Tensor idx, int maxNewTokens)
    {
        for (int i = 0; i < maxNewTokens; i++)
        {
            var (logits, _) = Forward(idx);
            logits = logits.select(1, -1);
            var probs = functional.softmax(logits, -1);
            Tensor idxNext = torch.multinomial(probs, 1);
            idx = torch.cat(new[] { idx, idxNext }, 1);
        }
        return idx;
    }
}

public class Encoding
{
    private readonly Dictionary<char, short> Encoder;
    private readonly Dictionary<short, char> Decoder;


    public Encoding(char[] chars)
    {
        Encoder = new Dictionary<char, short>();
        Decoder = new Dictionary<short, char>();
        for (short i = 0; i < chars.Length; i++)
        {
            Encoder.Add(chars[i], i);
            Decoder.Add(i, chars[i]);
        }
    }
    private short Encode(char ch)
    {
        return Encoder[ch];
    }

    private List<short> Encode(IEnumerable<char> chars)
    {
        return chars.Select(Encode).ToList();
    }

    public List<short> Encode(string chars)
    {
        return chars.Select(Encode).ToList();
    }

    private char Decode(short val)
    {
        return Decoder[val];
    }

    public string Decode(IEnumerable<short> vals)
    {
        return new string(vals.Select(Decode).ToArray());
    }
}