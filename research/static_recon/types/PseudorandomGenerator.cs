using Factory;
using Factory.Pools;
using FixMath;

[Serializable(1)]
public class PseudorandomGenerator : IReusable
{
	private ulong _x;

	private ulong _w;

	private static ulong _s = 13091206342165455529uL;

	public ulong Seed
	{
		get
		{
			return _x;
		}
		set
		{
			_x = value;
			_w = value;
		}
	}

	public void Reset()
	{
		_x = 0uL;
		_w = 0uL;
	}

	public int Int()
	{
		return NextInt();
	}

	public int Int(int max)
	{
		if (max <= 0)
		{
			return 0;
		}
		return (int)((uint)NextInt() % max);
	}

	public ulong ULong()
	{
		return NextULong();
	}

	public bool Bool()
	{
		return (NextInt() & 1) == 1;
	}

	public Fix64 Fix64()
	{
		long num = NextInt();
		num <<= 1;
		if (num < 0)
		{
			num = -num + 1;
		}
		return FixMath.Fix64.FromRaw(num);
	}

	public Fix64 Fix64(Fix64 max)
	{
		return Fix64() * max;
	}

	public override string ToString()
	{
		return $"PseudorandomGenerator[x={_x}, w={_w}]";
	}

	private ulong NextULong()
	{
		_x *= _x;
		_w += _s;
		_x += _w;
		_x = (_x >> 32) | (_x << 32);
		return _x;
	}

	private int NextInt()
	{
		return (int)NextULong();
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
