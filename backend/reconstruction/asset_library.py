"""Original reusable GLB furniture library and composed sample home.

Every asset in ``backend/assets/models`` is generated from the primitives in
this file.  No third-party model or texture is embedded, so the library can be
used by the sample studio and future photo-to-asset matching without licence
ambiguity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import trimesh


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
MODEL_ROOT = ASSET_ROOT / "models"

Color = List[int]
from .accessory_assets import ACCESSORIES, parts


class AssetLibrary:
    """Creates, stores, and instances original interior assets as GLB files."""

    ASSETS = {
        "bed": "Minimal wooden platform bed with frame, headboard, legs, and mattress only",
        "side_table": "Timber bedside table with a drawer and four legs",
        "tv_console": "Wall-facing television and low media console",
        "window": "Framed double-pane window with blue glass",
        "curtain": "Independent softly draped curtain panel",
        "floor_lamp": "Brass floor lamp with a warm fabric shade",
        "ceiling_lamp": "Ceiling pendant light",
        "desk": "Compact timber work desk",
        "chair": "Upholstered desk chair",
        "rug": "Soft area rug",
        "plant": "Low-poly potted house plant",
        "bookshelf": "Freestanding timber bookshelf with five open shelves",
        "book": "Closed hardback book with covers, spine, and page block",
        "book_upright": "Individual upright hardback with separate covers and spine",
        "book_paperback": "Thin individual paperback with flexible-looking covers",
        "cabinet": "Low two-door timber storage cabinet",
        "wardrobe": "Full-height two-door wooden wardrobe",
        "coffee_table": "Rounded timber coffee table with four legs",
        "sofa": "Upholstered two-seat sofa with separate back and arms",
        "wall_shelf": "Wall-mounted timber shelf with metal brackets",
        "mirror": "Framed wall mirror",
        "monitor": "Desktop display with screen, stand and base",
        "dresser": "Four-drawer timber dresser",
    }
    ASSETS.update({name:'Original multi-part '+name.replace('_',' ') for name in ACCESSORIES})
    REVISION = 8

    @classmethod
    def ensure_generated(cls) -> Dict[str, Path]:
        """Write each asset once to a stable reusable library directory."""
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = ASSET_ROOT / "asset_manifest.json"
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current_manifest = {}
        rebuild = current_manifest.get("revision") != cls.REVISION
        builders: Dict[str, Callable[[], trimesh.Scene]] = {
            "bed": cls._bed,
            "side_table": cls._side_table,
            "tv_console": cls._tv_console,
            "window": cls._window,
            "curtain": cls._curtain_asset,
            "floor_lamp": cls._floor_lamp,
            "ceiling_lamp": cls._ceiling_lamp,
            "desk": cls._desk,
            "chair": cls._chair,
            "rug": cls._rug,
            "plant": cls._plant,
            "bookshelf": cls._bookshelf,
            "book": cls._book,
            "book_upright": cls._book_upright,
            "book_paperback": cls._book_paperback,
            "cabinet": cls._cabinet,
            "wardrobe": cls._wardrobe,
            "coffee_table": cls._coffee_table,
            "sofa": cls._sofa,
            "wall_shelf": cls._wall_shelf,
            "mirror": cls._mirror,
            "monitor": cls._monitor,
            "dresser": cls._dresser,
        }
        paths: Dict[str, Path] = {}
        builders.update({name:(lambda name=name:cls._accessory(name)) for name in ACCESSORIES})
        for name, builder in builders.items():
            path = MODEL_ROOT / f"{name}.glb"
            if rebuild or not path.exists():
                path.write_bytes(builder().export(file_type="glb"))
            paths[name] = path
        (ASSET_ROOT / "asset_manifest.json").write_text(
            json.dumps({"revision": cls.REVISION, "format": "glb", "source": "original procedural assets", "assets": cls.ASSETS}, indent=2),
            encoding="utf-8",
        )
        return paths

    @classmethod
    def _accessory(cls,label):
        scene=trimesh.Scene()
        for i,(name,shape,size,center,color) in enumerate(parts(label)):
            if shape=='box': cls._box(scene,f'{name}_{i}',size,center,color)
            elif shape=='sphere': cls._sphere(scene,f'{name}_{i}',size,center,color)
            else: cls._cylinder(scene,f'{name}_{i}',size[0],size[1],center,color)
        return scene

    @staticmethod
    def _box(scene: trimesh.Scene, name: str, size: Sequence[float], center: Sequence[float], color: Color) -> None:
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @staticmethod
    def _cylinder(scene: trimesh.Scene, name: str, radius: float, height: float, center: Sequence[float], color: Color) -> None:
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=20)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        mesh.apply_translation(center)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @staticmethod
    def _cone(scene: trimesh.Scene, name: str, radius: float, height: float, center: Sequence[float], color: Color) -> None:
        mesh = trimesh.creation.cone(radius=radius, height=height, sections=20)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        mesh.apply_translation(center)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @staticmethod
    def _sphere(scene: trimesh.Scene, name: str, radii: Sequence[float], center: Sequence[float], color: Color) -> None:
        """A soft rounded primitive for upholstery, pillows, and bevel-like forms."""
        mesh = trimesh.creation.uv_sphere(count=[24, 16])
        mesh.apply_scale(radii)
        mesh.apply_translation(center)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @staticmethod
    def _rounded_box(scene: trimesh.Scene, name: str, size: Sequence[float], center: Sequence[float], color: Color) -> None:
        """Low-cost superellipsoid: rectangular overall form with softened edges."""
        mesh = trimesh.creation.uv_sphere(count=[28, 18])
        vertices = np.asarray(mesh.vertices)
        vertices = np.sign(vertices) * np.power(np.abs(vertices), 0.32)
        mesh.vertices = vertices * (np.asarray(size, dtype=float) / 2)
        mesh.apply_translation(center)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @staticmethod
    def _box_rotated_y(scene: trimesh.Scene, name: str, size: Sequence[float], center: Sequence[float], angle: float, color: Color) -> None:
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
        mesh.apply_translation(center)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @staticmethod
    def _fabric(scene: trimesh.Scene, name: str, width: float, depth: float, height: float, center: Sequence[float], color: Color, folds: float = 0.045) -> None:
        """Create a lightweight, genuinely wavy cloth surface for bedding/rugs."""
        columns, rows = 30, 22
        vertices = []
        for row in range(rows):
            v = row / (rows - 1)
            for column in range(columns):
                u = column / (columns - 1)
                x = (u - 0.5) * width
                z = (v - 0.5) * depth
                falloff = (1 - abs(2 * u - 1) ** 2) * (1 - abs(2 * v - 1) ** 2)
                y = height + folds * falloff * (np.sin(u * np.pi * 8) + 0.55 * np.sin(v * np.pi * 6 + u * 1.8))
                vertices.append([x + center[0], y + center[1], z + center[2]])
        faces = []
        for row in range(rows - 1):
            for column in range(columns - 1):
                start = row * columns + column
                faces.extend([[start, start + columns, start + 1], [start + 1, start + columns, start + columns + 1]])
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @staticmethod
    def _curtain(scene: trimesh.Scene, name: str, width: float, height: float, center: Sequence[float], color: Color) -> None:
        """A vertical draped panel with a small sinusoidal wave across its face."""
        columns, rows = 20, 24
        vertices = []
        for row in range(rows):
            for column in range(columns):
                u, v = column / (columns - 1), row / (rows - 1)
                vertices.append([
                    center[0] + (u - 0.5) * width,
                    center[1] + (v - 0.5) * height,
                    center[2] + np.sin(u * np.pi * 5) * 0.035,
                ])
        faces = []
        for row in range(rows - 1):
            for column in range(columns - 1):
                start = row * columns + column
                faces.extend([[start, start + columns, start + 1], [start + 1, start + columns, start + columns + 1]])
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)

    @classmethod
    def _bed(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        wood, dark, mattress = [151, 91, 48, 255], [104, 60, 34, 255], [239, 235, 225, 255]
        # A conventional platform construction: distinct rails, four legs,
        # a straight headboard and one rectangular mattress. Bedding is not
        # embedded; Gemini emits visible pillows/blankets as separate objects.
        cls._box(scene, "Side_Rail_Left", [.10, .28, 2.12], [-1.02, .33, 0], wood)
        cls._box(scene, "Side_Rail_Right", [.10, .28, 2.12], [1.02, .33, 0], wood)
        cls._box(scene, "Foot_Rail", [2.14, .30, .10], [0, .34, 1.01], wood)
        cls._box(scene, "Head_Rail", [2.14, .30, .10], [0, .34, -1.01], wood)
        for number, (x, z) in enumerate(((-1.0,-1.0),(-1.0,1.0),(1.0,-1.0),(1.0,1.0))):
            cls._box(scene, f"Leg_{number}", [.12, .54, .12], [x, .27, z], dark)
        cls._box(scene, "Headboard", [2.14, .66, .10], [0, .88, -1.04], wood)
        cls._rounded_box(scene, "Mattress", [1.98, .24, 1.94], [0, .56, -.02], mattress)
        return scene

    @classmethod
    def _side_table(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        wood, dark = [154, 98, 61, 255], [82, 51, 35, 255]
        cls._box(scene, "Cabinet", [0.66, 0.48, 0.52], [0, 0.56, 0], wood)
        cls._box(scene, "Drawer", [0.56, 0.18, 0.04], [0, 0.63, 0.25], [177, 116, 73, 255])
        cls._cylinder(scene, "Handle", 0.025, 0.11, [0, 0.63, 0.285], dark)
        for number, (x, z) in enumerate(((-.25, -.18), (-.25, .18), (.25, -.18), (.25, .18))):
            cls._cylinder(scene, f"Leg_{number}", 0.038, 0.55, [x, 0.275, z], dark)
        return scene

    @classmethod
    def _tv_console(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        cls._box(scene, "Console", [1.92, 0.58, 0.54], [0, 0.275, 0], [69, 59, 53, 255])
        cls._box(scene, "Console_Shelf", [1.65, 0.04, 0.42], [0, 0.28, 0.02], [112, 85, 67, 255])
        cls._box(scene, "Screen", [1.60, 0.96, 0.11], [0, 1.18, -0.12], [18, 29, 38, 255])
        cls._box(scene, "Screen_Glow", [1.46, 0.82, 0.024], [0, 1.18, -0.18], [37, 65, 78, 255])
        cls._box(scene, "TV_Stand", [0.45, 0.16, 0.24], [0, 0.70, -0.03], [25, 27, 29, 255])
        return scene

    @classmethod
    def _window(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        frame, glass = [242, 241, 236, 255], [104, 181, 211, 255]
        cls._box(scene, "Glass_Left", [0.89, 1.35, 0.035], [-0.46, 1.45, 0], glass)
        cls._box(scene, "Glass_Right", [0.89, 1.35, 0.035], [0.46, 1.45, 0], glass)
        cls._box(scene, "Frame_Top", [2.00, 0.09, 0.10], [0, 2.17, 0], frame)
        cls._box(scene, "Frame_Bottom", [2.00, 0.09, 0.10], [0, 0.73, 0], frame)
        cls._box(scene, "Frame_Left", [0.09, 1.53, 0.10], [-1.00, 1.45, 0], frame)
        cls._box(scene, "Frame_Right", [0.09, 1.53, 0.10], [1.00, 1.45, 0], frame)
        cls._box(scene, "Frame_Centre", [0.07, 1.38, 0.08], [0, 1.45, 0], frame)
        return scene

    @classmethod
    def _curtain_asset(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        cls._curtain(scene, "Draped_Panel", 1.0, 2.15, [0, 1.08, 0], [226, 220, 209, 255])
        cls._box(scene, "Curtain_Rod", [1.08, .035, .035], [0, 2.20, 0], [85, 81, 76, 255])
        return scene

    @classmethod
    def _floor_lamp(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        brass, shade = [186, 143, 67, 255], [239, 213, 159, 255]
        cls._cylinder(scene, "Base", 0.20, 0.06, [0, 0.03, 0], brass)
        cls._cylinder(scene, "Stem", 0.035, 1.42, [0, 0.74, 0], brass)
        cls._cone(scene, "Shade", 0.30, 0.42, [0, 1.63, 0], shade)
        return scene

    @classmethod
    def _ceiling_lamp(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        cls._cylinder(scene, "Cord", 0.018, 0.62, [0, 2.69, 0], [45, 45, 49, 255])
        cls._cone(scene, "Shade", 0.28, 0.34, [0, 2.32, 0], [238, 193, 89, 255])
        cls._cylinder(scene, "Bulb", 0.10, 0.14, [0, 2.13, 0], [255, 242, 181, 255])
        return scene

    @classmethod
    def _desk(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        wood, dark = [181, 119, 70, 255], [65, 45, 36, 255]
        cls._box(scene, "Top", [1.45, 0.10, 0.62], [0, 0.77, 0], wood)
        for number, (x, z) in enumerate(((-.62, -.23), (-.62, .23), (.62, -.23), (.62, .23))):
            cls._box(scene, f"Leg_{number}", [0.07, 0.75, 0.07], [x, 0.375, z], dark)
        cls._box(scene, "Drawer", [0.42, 0.14, 0.05], [0.32, 0.64, 0.33], [151, 93, 55, 255])
        return scene

    @classmethod
    def _chair(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        fabric, metal = [79, 131, 147, 255], [45, 50, 54, 255]
        cls._rounded_box(scene, "Seat", [0.60, 0.14, 0.56], [0, 0.51, 0], fabric)
        cls._rounded_box(scene, "Back", [0.58, 0.62, 0.11], [0, 0.88, -0.25], fabric)
        cls._cylinder(scene, "Column", 0.045, 0.46, [0, 0.23, 0], metal)
        for side in (-1, 1):
            cls._box(scene, f"Arm_Support_{side}", [.04, .30, .04], [side*.34, .69, -.03], metal)
            cls._box(scene, f"Armrest_{side}", [.10, .06, .38], [side*.34, .82, -.02], metal)
        for number in range(5):
            angle = number * np.pi * 2 / 5
            x, z = np.sin(angle) * .25, np.cos(angle) * .25
            cls._box_rotated_y(scene, f"Base_Arm_{number}", [.07, .045, .54], [x, .06, z], angle, metal)
            cls._sphere(scene, f"Wheel_{number}", [.055, .055, .04], [np.sin(angle)*.51, .045, np.cos(angle)*.51], metal)
        return scene

    @classmethod
    def _rug(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        cls._box(scene, "Rug", [2.60, 0.035, 1.85], [0, 0.018, 0], [198, 156, 93, 255])
        cls._box(scene, "Rug_Inset", [2.20, 0.012, 1.48], [0, 0.045, 0], [49, 104, 124, 255])
        return scene

    @classmethod
    def _monitor(cls) -> trimesh.Scene:
        scene=trimesh.Scene()
        cls._box(scene,'Display',[.60,.34,.035],[0,.35,0],[25,28,32,255])
        cls._box(scene,'Screen',[.56,.30,.008],[0,.35,.022],[42,62,75,255])
        cls._box(scene,'Stand',[.05,.20,.04],[0,.10,0],[65,68,72,255])
        cls._box(scene,'Base',[.27,.02,.18],[0,.01,0],[65,68,72,255])
        return scene

    @classmethod
    def _dresser(cls) -> trimesh.Scene:
        scene=trimesh.Scene()
        cls._box(scene,'Case',[1.1,.85,.45],[0,.425,0],[151,94,54,255])
        for i in range(4):
            cls._box(scene,f'Drawer_{i}',[1.05,.195,.025],[0,.11+i*.21,.235],[112,72,45,255])
            cls._box(scene,f'Handle_{i}',[.20,.015,.035],[0,.11+i*.21,.26],[65,68,72,255])
        return scene

    @classmethod
    def _plant(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        cls._cone(scene, "Pot", 0.20, 0.34, [0, 0.17, 0], [194, 103, 60, 255])
        cls._cylinder(scene, "Stem", 0.035, 0.66, [0, 0.61, 0], [61, 107, 58, 255])
        for number, (x, z) in enumerate(((-.16, 0), (.16, 0), (0, -.16), (0, .16))):
            cls._box(scene, f"Leaf_{number}", [0.12, 0.42, 0.08], [x, 0.75, z], [59, 139, 81, 255])
        return scene

    @classmethod
    def _bookshelf(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        wood, dark = [151, 94, 54, 255], [82, 51, 35, 255]
        width, depth, height, panel = 1.20, .38, 1.90, .055
        cls._box(scene, "Back", [width, height, .035], [0, height / 2, -depth / 2], dark)
        for side in (-1, 1):
            cls._box(scene, f"Side_{side}", [panel, height, depth], [side * (width / 2 - panel / 2), height / 2, 0], wood)
        for index, y in enumerate((.04, .48, .92, 1.36, 1.86)):
            cls._box(scene, f"Shelf_{index}", [width, panel, depth], [0, y, 0], wood)
        return scene

    @classmethod
    def _book(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        cls._box(scene, "Pages", [.21, .024, .155], [0, .015, 0], [232, 224, 201, 255])
        cls._box(scene, "Front_Cover", [.225, .003, .17], [0, .0285, 0], [102, 75, 126, 255])
        cls._box(scene, "Back_Cover", [.225, .003, .17], [0, .0015, 0], [102, 75, 126, 255])
        cls._box(scene, "Spine", [.008, .03, .17], [-.1085, .015, 0], [83, 58, 109, 255])
        return scene

    @classmethod
    def _book_upright(cls):
        scene=cls._book()
        scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,0,1]))
        scene.apply_translation([0,-scene.bounds[0,1],0])
        return scene

    @classmethod
    def _book_paperback(cls):
        scene=cls._book()
        scene.apply_transform(np.diag([.85,.45,.9,1]))
        return scene

    @classmethod
    def _cabinet(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        wood, dark, metal = [151, 94, 54, 255], [93, 56, 34, 255], [58, 61, 63, 255]
        cls._rounded_box(scene, "Body", [1.20, .82, .46], [0, .51, 0], wood)
        for side in (-1, 1):
            cls._box(scene, f"Door_{side}", [.55, .68, .025], [side * .29, .53, .245], dark)
            cls._cylinder(scene, f"Handle_{side}", .018, .13, [side * .09, .54, .275], metal)
            cls._box(scene, f"Foot_{side}", [.12, .14, .30], [side * .42, .07, 0], dark)
        return scene

    @classmethod
    def _wardrobe(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        wood, dark, metal = [134, 85, 52, 255], [76, 48, 33, 255], [70, 73, 75, 255]
        cls._box(scene, "Body", [1.55, 2.10, .58], [0, 1.05, 0], wood)
        for side in (-1, 1):
            cls._box(scene, f"Door_{side}", [.73, 1.96, .028], [side * .385, 1.07, .305], dark)
            cls._cylinder(scene, f"Handle_{side}", .016, .22, [side * .08, 1.08, .34], metal)
        return scene

    @classmethod
    def _coffee_table(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        wood, dark = [164, 105, 62, 255], [80, 51, 35, 255]
        cls._rounded_box(scene, "Top", [1.18, .10, .70], [0, .43, 0], wood)
        for number, (x, z) in enumerate(((-.48, -.24), (-.48, .24), (.48, -.24), (.48, .24))):
            cls._cylinder(scene, f"Leg_{number}", .035, .38, [x, .19, z], dark)
        return scene

    @classmethod
    def _sofa(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        fabric, shadow = [107, 124, 133, 255], [79, 93, 101, 255]
        cls._rounded_box(scene, "Base", [1.95, .36, .82], [0, .25, 0], shadow)
        for side in (-1, 1):
            cls._rounded_box(scene, f"Seat_{side}", [.84, .18, .62], [side * .44, .50, .08], fabric)
            cls._rounded_box(scene, f"Back_Cushion_{side}", [.84, .55, .20], [side * .44, .78, -.29], fabric)
            cls._rounded_box(scene, f"Arm_{side}", [.18, .49, .82], [side * .91, .48, 0], shadow)
        return scene

    @classmethod
    def _wall_shelf(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        cls._rounded_box(scene, "Shelf", [1.05, .07, .28], [0, .04, 0], [151, 94, 54, 255])
        for side in (-1, 1):
            cls._box(scene, f"Bracket_{side}", [.045, .25, .20], [side * .34, -.10, -.04], [66, 68, 70, 255])
        return scene

    @classmethod
    def _mirror(cls) -> trimesh.Scene:
        scene = trimesh.Scene()
        frame, glass = [91, 58, 39, 255], [153, 190, 199, 255]
        cls._box(scene, "Glass", [.82, 1.12, .018], [0, .60, 0], glass)
        cls._box(scene, "Frame_Top", [.94, .07, .055], [0, 1.20, 0], frame)
        cls._box(scene, "Frame_Bottom", [.94, .07, .055], [0, 0, 0], frame)
        cls._box(scene, "Frame_Left", [.07, 1.27, .055], [-.47, .60, 0], frame)
        cls._box(scene, "Frame_Right", [.07, 1.27, .055], [.47, .60, 0], frame)
        return scene

    @classmethod
    def add_instance(
        cls,
        scene: trimesh.Scene,
        asset: str,
        group: str,
        position: Sequence[float],
        rotation_y: float = 0,
        scale: float = 1.0,
        parent: str = "Furnishings",
    ) -> None:
        """Add a named instance of a stored GLB asset to another GLB scene."""
        asset_path = cls.ensure_generated()[asset]
        source = trimesh.load(asset_path, force="scene")
        scene.graph.update(frame_to=group, frame_from=parent, matrix=np.eye(4))
        transform = trimesh.transformations.rotation_matrix(rotation_y, [0, 1, 0])
        transform = transform @ trimesh.transformations.scale_matrix(scale)
        transform[:3, 3] = position
        for node in source.graph.nodes_geometry:
            local_transform, geometry_name = source.graph[node]
            mesh = source.geometry[geometry_name].copy()
            mesh.apply_transform(local_transform)
            mesh.apply_transform(transform)
            node_name = f"{group}_{node}"
            scene.add_geometry(mesh, node_name=node_name, geom_name=node_name, parent_node_name=group)

    @classmethod
    def create_sample_home(cls, output_glb_path: str) -> str:
        """Assemble the reusable assets in a colourful, walkable sample home."""
        cls.ensure_generated()
        scene = trimesh.Scene()
        identity = np.eye(4)
        scene.graph.update(frame_to="Sample_Home", frame_from="world", matrix=identity)
        scene.graph.update(frame_to="Architecture", frame_from="Sample_Home", matrix=identity)
        scene.graph.update(frame_to="Furnishings", frame_from="Sample_Home", matrix=identity)

        def architecture_box(name: str, size: Sequence[float], center: Sequence[float], color: Color) -> None:
            mesh = trimesh.creation.box(extents=size)
            mesh.apply_translation(center)
            mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
            scene.add_geometry(mesh, node_name=name, geom_name=name, parent_node_name="Architecture")

        # The open front keeps the camera entry clear; all furniture sits inside
        # an 8 m by 10 m room extending away from the initial camera.
        architecture_box("Oak_Floor", [8.0, 0.08, 10.0], [0, -0.04, -4.85], [153, 113, 77, 255])
        architecture_box("Back_Wall", [8.0, 3.0, 0.12], [0, 1.5, -9.80], [230, 218, 201, 255])
        architecture_box("Left_Wall", [0.12, 3.0, 10.0], [-3.94, 1.5, -4.85], [219, 210, 197, 255])
        architecture_box("Right_Wall", [0.12, 3.0, 10.0], [3.94, 1.5, -4.85], [219, 210, 197, 255])
        architecture_box("Ceiling", [8.0, 0.08, 10.0], [0, 3.0, -4.85], [246, 243, 238, 255])
        architecture_box("Accent_Wall", [2.3, 2.7, 0.04], [-2.65, 1.42, -9.70], [93, 122, 125, 255])

        cls.add_instance(scene, "bed", "Bed", [-1.40, 0, -7.65])
        # The bedside space is reserved for the locally hosted, detailed CC0
        # Poly Haven nightstand loaded by the web viewer.
        cls.add_instance(scene, "tv_console", "TV_Console", [2.65, 0, -8.95])
        cls.add_instance(scene, "window", "Window", [-3.86, 0, -5.40], np.pi / 2)
        cls.add_instance(scene, "window", "Window_Second", [-3.86, 0, -7.55], np.pi / 2)
        cls.add_instance(scene, "floor_lamp", "Floor_Lamp", [1.25, 0, -5.90])
        cls.add_instance(scene, "ceiling_lamp", "Ceiling_Lamp", [0.15, 0, -4.35])
        cls.add_instance(scene, "desk", "Desk", [-2.55, 0, -3.15], np.pi / 2)
        cls.add_instance(scene, "chair", "Chair", [-1.75, 0, -3.15], np.pi / 2)
        cls.add_instance(scene, "rug", "Rug", [0.40, 0, -4.35])
        cls.add_instance(scene, "plant", "Plant", [3.15, 0, -5.65])

        output = Path(output_glb_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(scene.export(file_type="glb"))
        return str(output)
