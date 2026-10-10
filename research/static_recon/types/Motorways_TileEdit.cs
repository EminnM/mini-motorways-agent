using System.Collections.Generic;
using Factory;
using Factory.Pools;
using Motorways.Models;
using Server;

namespace Motorways;

[Serializable(1)]
public abstract class TileEdit : IReusable
{
	protected static Diagnostics.Log.Channel Log = Diagnostics.Log.OpenChannel("TileEdit");

	[Dependency]
	protected GameBehaviourModel _behaviour;

	public bool CanApplyToSimulation { get; set; } = true;

	public virtual void Reset()
	{
		CanApplyToSimulation = true;
	}

	public virtual IEnumerable<Tile> GetAffectedTiles(ITilemap tilemap)
	{
		yield break;
	}

	public virtual bool ApplyToAffectedTile(Tile tile)
	{
		return false;
	}

	public virtual IEnumerable<Motorway> GetAffectedMotorways(ITilemap tilemap)
	{
		yield break;
	}

	public virtual bool ApplyToAffectedMotorway(Motorway motorway)
	{
		return false;
	}

	public virtual void ApplyToSimulation(ISimulation simulation)
	{
	}

	public bool ApplyToTilemap(ITilemap tilemap)
	{
		Log.Info("Applying {0} to tilemap.", this);
		bool flag = true;
		foreach (Motorway affectedMotorway in GetAffectedMotorways(tilemap))
		{
			Log.Info("Applying to motorway {0}.", affectedMotorway);
			flag = ApplyToAffectedMotorway(affectedMotorway) && flag;
		}
		foreach (Tile affectedTile in GetAffectedTiles(tilemap))
		{
			Log.Info("Applying to tile {0}.", affectedTile);
			flag = ApplyToAffectedTile(affectedTile) && flag;
		}
		return flag;
	}

	public abstract bool ApplyToUpgradeDatabase(UpgradeDatabase upgradeDatabase, ITilemap tilemap);
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
