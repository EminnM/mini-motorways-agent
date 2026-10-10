using System;
using System.Collections.Generic;
using Factory;
using UnityEngine;

public static class Random
{
	private static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("Random");

	private static System.Random _randomSource = new System.Random(Environment.TickCount);

	private static PseudorandomGenerator _simulationSeedGenerator = null;

	public static void SetSimulationSeed(uint seed, IScope scope)
	{
		Log.Info("Set simulation seed {0}.", seed);
		if (_simulationSeedGenerator == null)
		{
			_simulationSeedGenerator = scope.Get<PseudorandomGenerator>();
		}
		_simulationSeedGenerator.Seed = seed;
	}

	public static float Float()
	{
		return (float)NextDouble();
	}

	public static float Float(int seed)
	{
		return (float)new System.Random(seed).NextDouble();
	}

	public static double Double()
	{
		return NextDouble();
	}

	public static float Float(float max)
	{
		return (float)NextDouble() * max;
	}

	public static int Int()
	{
		return NextInt();
	}

	public static int Int(int max)
	{
		if (max == 0)
		{
			return 0;
		}
		return NextInt() % max;
	}

	public static float Range(float low, float high)
	{
		return low + (high - low) * (float)NextDouble();
	}

	public static int Range(int low, int high)
	{
		int num = high - low;
		if (num == 0)
		{
			return 0;
		}
		return low + NextInt() % num;
	}

	public static bool Bool()
	{
		return NextDouble() < 0.5;
	}

	public static object Select(params object[] objects)
	{
		return objects[NextInt() % objects.Length];
	}

	public static T AnyItem<T>(T[] items)
	{
		if (items.Length == 0)
		{
			return default(T);
		}
		return items[NextInt() % items.Length];
	}

	public static T AnyItem<T>(List<T> items)
	{
		if (items.Count == 0)
		{
			return default(T);
		}
		return items[NextInt() % items.Count];
	}

	public static Vector2 Vector2Normalized()
	{
		return new Vector2(Range(-1f, 1f), Range(-1f, 1f)).normalized;
	}

	public static Vector3 Vector3Normalized()
	{
		return new Vector3(Range(-1f, 1f), Range(-1f, 1f), Range(-1f, 1f)).normalized;
	}

	public static void ShuffleList<T>(List<T> list)
	{
		list.Shuffle();
	}

	public static void Shuffle<T>(this List<T> list)
	{
		int num = list.Count;
		while (num > 1)
		{
			num--;
			int index = Int(num + 1);
			T value = list[index];
			list[index] = list[num];
			list[num] = value;
		}
	}

	public static void Shuffle<T>(this List<T> list, PseudorandomGenerator rand)
	{
		int num = list.Count;
		while (num > 1)
		{
			num--;
			int index = rand.Int(num + 1);
			T value = list[index];
			list[index] = list[num];
			list[num] = value;
		}
	}

	public static uint NextSimulationSeed()
	{
		if (_simulationSeedGenerator == null)
		{
			return 0u;
		}
		return (uint)(_simulationSeedGenerator.Int() + 1);
	}

	private static int NextInt()
	{
		return _randomSource.Next();
	}

	private static double NextDouble()
	{
		return _randomSource.NextDouble();
	}

	private static int RandomComparison<T>(T a, T b)
	{
		if (NextInt() % 2 == 0)
		{
			return -1;
		}
		return 1;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
