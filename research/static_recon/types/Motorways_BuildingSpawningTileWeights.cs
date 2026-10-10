using System.Collections.Generic;
using FixMath;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace Motorways;

public class BuildingSpawningTileWeights
{
	public Dictionary<Vector3Int, Fix64> weights;

	public BuildingSpawningTileWeights(Dictionary<Vector3Int, Fix64> weights)
	{
		this.weights = weights;
	}

	public BuildingSpawningTileWeights(Tilemap tilemap)
	{
		weights = new Dictionary<Vector3Int, Fix64>();
		foreach (Vector3Int item in tilemap.cellBounds.allPositionsWithin)
		{
			if (tilemap.GetTile(item) != null)
			{
				WeightTile weightTile = tilemap.GetTile(item) as WeightTile;
				if (weightTile != null)
				{
					weights[item] = (Fix64)weightTile.tileWeight;
				}
				else
				{
					weights[item] = Fix64.One;
				}
			}
		}
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
