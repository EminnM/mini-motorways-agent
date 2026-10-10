using Factory;
using Factory.Pools;

namespace Server;

[Serializable(1)]
public abstract class Command : IReusable
{
	public static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("Server.Command");

	private int _frameIndex = -1;

	public int FrameIndex
	{
		get
		{
			return _frameIndex;
		}
		set
		{
			_frameIndex = value;
		}
	}

	public abstract void Execute(ISimulation simulation);

	public virtual void Reset()
	{
		_frameIndex = -1;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
