using UnityEngine;
using UnityEngine.Tilemaps;

namespace Motorways;

[CreateAssetMenu(fileName = "New Density Tile", menuName = "Tiles/Spawn Density Tile")]
public class DensityTile : TileBase
{
	public Sprite sprite;

	public Color color;

	public DensityGroup group;

	public override void GetTileData(Vector3Int position, UnityEngine.Tilemaps.ITilemap tilemap, ref TileData tileData)
	{
		base.GetTileData(position, tilemap, ref tileData);
		tileData.flags = TileFlags.LockColor;
		tileData.sprite = sprite;
		tileData.color = color;
	}
}
You are not using the latest version of the tool, please update.
Latest version is '11.1.0.9782' (yours is '9.1.0.7988')
