namespace Shared;

public class TokenEncoder
{
    private readonly Dictionary<char, short> _encoder;
    private readonly Dictionary<short, char> _decoder;

    // Timestamp: 10:00
    public TokenEncoder(char[] chars)
    {
        _encoder = new Dictionary<char, short>();
        _decoder = new Dictionary<short, char>();
        for (short i = 0; i < chars.Length; i++)
        {
            _encoder.Add(chars[i], i);
            _decoder.Add(i, chars[i]);
        }
    }
    private short Encode(char ch)
    {
        return _encoder[ch];
    }

    public List<short> Encode(string chars)
    {
        return chars.Select(Encode).ToList();
    }

    public char Decode(short val)
    {
        return _decoder[val];
    }

    public string Decode(IEnumerable<short> vals)
    {
        return new string(vals.Select(Decode).ToArray());
    }
}