using System.Collections.Generic;
using Factory;
using UnityEngine;

namespace Motorways;

public class AddRoadEdit : TileEdit
{
	private bool _isStartingPassage;

	private bool _isJoiningPassages;

	private UpgradeType _passageType;

	[Serialize(true, null)]
	public Vector2Int OriginCoordinates { get; private set; }

	[Serialize(true, null)]
	public TileDirection OriginDirection { get; private set; }

	public Vector2Int DestinationCoordinates => TileUtilities.GetAdjacentCoordinates(OriginCoordinates, OriginDirection);

	public TileDirection DestinationDirection => TileUtilities.GetOppositeDirection(OriginDirection);

	public override IEnumerable<Tile> GetAffectedTiles(ITilemap tilemap)
	{
		yield return tilemap.GetOrCreateTile(OriginCoordinates);
		yield return tilemap.GetOrCreateTile(DestinationCoordinates);
	}

	public override bool ApplyToAffectedTile(Tile tile)
	{
		TileDirection tileDirection = TileDirection.None;
		if (tile.Coordinates == OriginCoordinates)
		{
			tileDirection = OriginDirection;
		}
		else if (tile.Coordinates == DestinationCoordinates)
		{
			tileDirection = DestinationDirection;
		}
		if (tileDirection != TileDirection.None)
		{
			if (Roundabout.IsTileCenterOfRoundabout(tile))
			{
				return true;
			}
			tile.SetNodeState(new RoadTileNode(tileDirection), RoadState.Pending);
			return true;
		}
		return false;
	}

	public override bool ApplyToUpgradeDatabase(UpgradeDatabase upgradeDatabase, ITilemap tilemap)
	{
		Tile orCreateTile = tilemap.GetOrCreateTile(OriginCoordinates);
		Tile orCreateTile2 = tilemap.GetOrCreateTile(DestinationCoordinates);
		if (_isJoiningPassages)
		{
			upgradeDatabase.MothballUpgrade(_passageType);
			upgradeDatabase.ReleaseMothballedUpgrade(_passageType);
		}
		else if (_isStartingPassage)
		{
			upgradeDatabase.ConsumeUpgrade(_passageType);
		}
		int concreteCostForConnection = _behaviour.GetConcreteCostForConnection(orCreateTile, orCreateTile2);
		if (concreteCostForConnection > 0)
		{
			if (orCreateTile.GetTwoLaneRoadStateInDirection(OriginDirection) == RoadState.Mothballed || orCreateTile2.GetTwoLaneRoadStateInDirection(TileUtilities.GetOppositeDirection(OriginDirection)) == RoadState.Mothballed)
			{
				return upgradeDatabase.UnmothballUpgrade(UpgradeType.Concrete, concreteCostForConnection);
			}
			return upgradeDatabase.ConsumeUpgrade(UpgradeType.Concrete, concreteCostForConnection);
		}
		return true;
	}

	public override void Reset()
	{
		base.Reset();
		OriginCoordinates = default(Vector2Int);
		OriginDirection = TileDirection.North;
		_isStartingPassage = false;
		_isJoiningPassages = false;
		_passageType = UpgradeType.Concrete;
	}

	public static AddRoadEdit Create(IScope scope, ITilemap tilemap, Vector2Int originCoordinates, TileDirection direction, CityDefinition cityDefinition)
	{
		AddRoadEdit addRoadEdit = scope.Get<AddRoadEdit>();
		addRoadEdit.OriginCoordinates = originCoordinates;
		addRoadEdit.OriginDirection = direction;
		Vector2Int destinationCoordinates = addRoadEdit.DestinationCoordinates;
		UpgradeType passageType2;
		if (Passage.WillConnectionStartPassage(cityDefinition, tilemap, originCoordinates, destinationCoordinates, out var passageType))
		{
			addRoadEdit._isStartingPassage = true;
			addRoadEdit._passageType = passageType;
		}
		else if (Passage.WillConnectionJoinPassages(cityDefinition, tilemap, originCoordinates, destinationCoordinates, out passageType2))
		{
			addRoadEdit._isJoiningPassages = true;
			addRoadEdit._passageType = passageType2;
		}
		return addRoadEdit;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
