using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace Shared;
public static class TensorExtensions
{
    public static string ToFormattedString(this torch.Tensor tensor)
    {
        StringBuilder builder = new StringBuilder();

        builder.AppendLine();

        // Determine tensor type and appropriate conversion
        Func<torch.Tensor, string> getItem;
        switch (tensor.dtype)
        {
            case torch.ScalarType.Float32:
                getItem = t => t.item<float>().ToString("F2");
                break;
            case torch.ScalarType.Int64:
                getItem = t => t.item<long>().ToString();
                break;
            case torch.ScalarType.Int32:
                getItem = t => t.item<int>().ToString();
                break;
            // Add other types as needed
            default:
                getItem = t => t.ToString();
                break;
        }

        if (tensor.dim() == 1)
        {
            builder.Append("[");
            for (long i = 0; i < tensor.size(0); i++)
            {
                builder.Append(getItem(tensor[i]));
                if (i < tensor.size(0) - 1)
                    builder.Append(", ");
            }
            builder.Append("]");
        }
        else if (tensor.dim() == 2)
        {
            builder.Append("[");
            for (long i = 0; i < tensor.size(0); i++)
            {
                builder.Append("[");
                for (long j = 0; j < tensor.size(1); j++)
                {
                    builder.Append(getItem(tensor[i, j]));
                    if (j < tensor.size(1) - 1)
                        builder.Append(", ");
                }
                builder.Append("]");
                if (i < tensor.size(0) - 1)
                    builder.AppendLine(",");
                else if (i != tensor.size(0) - 1)
                    builder.AppendLine();
            }
            builder.Append("]");
        }
        else if (tensor.dim() == 3)
        {
            builder.Append("[");
            for (long i = 0; i < tensor.size(0); i++)
            {
                builder.AppendLine($"Slice {i}:");
                builder.Append("[");
                for (long j = 0; j < tensor.size(1); j++)
                {
                    builder.Append("[");
                    for (long k = 0; k < tensor.size(2); k++)
                    {
                        builder.Append(getItem(tensor[i, j, k]));
                        if (k < tensor.size(2) - 1)
                            builder.Append(", ");
                    }
                    builder.Append("]");
                    if (j < tensor.size(1) - 1)
                        builder.AppendLine(",");
                    else if (j != tensor.size(0) - 1)
                        builder.AppendLine();
                }
                builder.Append("]");
                if (i < tensor.size(0) - 1)
                    builder.AppendLine(",");
            }
            builder.AppendLine("]");
        }
        else
        {
            return "Tensors with more than 3 dimensions are not supported in this method.";
        }

        builder.AppendLine();

        return builder.ToString();
    }

    public static Tensor Softmax(this Tensor input, int dim = -1)
    {
        // Compute e^x_i for each element x_i in the input tensor.
        Tensor expTensor = input.exp();

        // Sum the exponentiated values along the specified dimension, returns single value.
        Tensor sumExp = expTensor.sum(dim, true);

        // Divide each exponentiated value by the sum of all exponentiated values.
        return expTensor / sumExp;
    }
}