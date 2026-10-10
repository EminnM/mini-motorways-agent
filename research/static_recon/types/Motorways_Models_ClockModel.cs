using Factory;
using FixMath;
using Server;

namespace Motorways.Models;

public class ClockModel : Model<ClockModel.Frame, IEmptyModelObserver>
{
	public class Frame : IFrame
	{
		public Fix64 time;

		public Fix64 expansionTime;

		public void Reset()
		{
			time = Fix64.Zero;
			expansionTime = Fix64.Zero;
		}

		public bool CloneInto(IFrame cloneState, IScope scope)
		{
			Frame obj = (Frame)cloneState;
			obj.time = time;
			obj.expansionTime = expansionTime;
			return true;
		}
	}

	public const double SecondsPerHour = 5.0 / 6.0;

	public const double SecondsPerWeek = 140.0;

	public bool isPaused;

	public bool expansionTimeManuallyPaused;

	public int Hour => (int)(long)(base.CurrentFrame.time / (Fix64)(5.0 / 6.0));

	public int Day => Hour / 24;

	public int Week => Hour / 168;

	public Fix64 Time => base.CurrentFrame.time;

	public Fix64 FractionalDays => base.CurrentFrame.time / (Fix64)20.0;

	public int ExpansionHour => (int)(long)(base.CurrentFrame.expansionTime / (Fix64)(5.0 / 6.0));

	public int ExpansionDay => ExpansionHour / 24;

	public int ExpansionWeek => ExpansionHour / 168;

	public Fix64 ExpansionTime => base.CurrentFrame.expansionTime;

	public static int SecondsToHours(Fix64 seconds)
	{
		return (int)(long)(seconds / (Fix64)(5.0 / 6.0));
	}

	public static Fix64 HoursToSeconds(int hours)
	{
		return (Fix64)((double)hours * (5.0 / 6.0));
	}

	public static int SecondsToDays(Fix64 seconds)
	{
		return (int)(long)(seconds / (Fix64)(5.0 / 6.0)) / 24;
	}

	public static Fix64 SecondsToFractionalDays(Fix64 seconds)
	{
		return seconds / (Fix64)(5.0 / 6.0) / (Fix64)24L;
	}

	public static Fix64 DaysToSeconds(int days)
	{
		return (Fix64)((double)days * (5.0 / 6.0) * 24.0);
	}

	public static Fix64 DaysToSeconds(Fix64 days)
	{
		return Fix64.FastMul(days, (Fix64)20.0);
	}

	public static int SecondsToWeeks(Fix64 seconds)
	{
		return (int)(long)(seconds / (Fix64)(5.0 / 6.0)) / 168;
	}

	public override void Reset()
	{
		base.Reset();
		isPaused = false;
	}

	public float GetInterpolatedTime(float alpha)
	{
		return (float)base.CurrentFrame.time + (float)(base.NextFrame.time - base.CurrentFrame.time) * alpha;
	}

	public Fix64 GetInterpolatedExpansionTime(float alpha)
	{
		return base.CurrentFrame.expansionTime + (base.NextFrame.expansionTime - base.CurrentFrame.expansionTime) * (Fix64)alpha;
	}

	public void SetExpansionTimeToDay(Fix64 expansionTimeDay)
	{
		base.CurrentFrame.expansionTime = DaysToSeconds(expansionTimeDay);
		base.NextFrame.expansionTime = base.CurrentFrame.expansionTime;
	}

	public ClockModel()
		: base(1)
	{
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
