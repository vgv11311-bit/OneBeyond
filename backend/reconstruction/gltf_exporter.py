"""Neutral scene-graph GLB exporter for single-photo room reconstruction.

The exporter has no code path that embeds the source image. Each resulting mesh
is an independent GLB node so the viewer can walk around it and a future editor
can select it by its hierarchy path.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .asset_library import AssetLibrary


class GLTFExporter:
    """Turn inferred room and furniture dimensions into a navigable scene tree."""

    NEUTRAL = [188, 192, 198, 255]
    DARK = [84, 92, 102, 255]
    LIGHT = [222, 225, 229, 255]
    FLOOR = [126, 133, 141, 255]

    @staticmethod
    def create_room_glb(
        output_glb_path: str,
        room_dimensions: Tuple[float, float, float],
        dense_surface_mesh: Optional[trimesh.Trimesh] = None,
        furniture_specs: Optional[List[Dict[str, Any]]] = None,
        point_cloud: Optional[np.ndarray] = None,
        point_colors: Optional[np.ndarray] = None,
        floor_color: Optional[List[float]] = None,
        wall_color: Optional[List[float]] = None,
        ceiling_color: Optional[List[float]] = None,
        camera_positions: Optional[List[np.ndarray]] = None,
        primary_image_path: Optional[str] = None,
    ) -> str:
        """Export simple spatial structure; photo pixels are deliberately unused."""
        width, length, height = room_dimensions
        # Preserve a user-provided floor-plan calibration instead of silently
        # snapping it back to the single-photo default envelope.
        width = float(np.clip(width, 2.0, 20.0))
        length = float(np.clip(length, 3.0, 25.0))
        height = float(np.clip(height, 2.2, 5.0))
        def surface_color(value: Optional[List[float]], fallback: List[int]) -> List[int]:
            if not value or len(value) < 3:
                return fallback
            values = np.asarray(value[:3], dtype=float)
            if float(values.max()) <= 1:
                values *= 255
            return [int(channel) for channel in np.clip(values, 24, 245)] + [255]
        floor_rgba = surface_color(floor_color, GLTFExporter.FLOOR)
        wall_rgba = surface_color(wall_color, GLTFExporter.LIGHT)
        ceiling_rgba = surface_color(ceiling_color, GLTFExporter.LIGHT)
        scene = trimesh.Scene()
        identity = np.eye(4)
        scene.graph.update(frame_to="Room", frame_from="world", matrix=identity)
        scene.graph.update(frame_to="Room_Envelope", frame_from="Room", matrix=identity)
        scene.graph.update(frame_to="Objects", frame_from="Room", matrix=identity)

        def add_box(name: str, extents: List[float], center: List[float], rgba: List[int], parent: str) -> None:
            mesh = trimesh.creation.box(extents=np.maximum(np.asarray(extents, dtype=float), 0.02))
            mesh.apply_translation(center)
            mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
            scene.add_geometry(mesh, node_name=name, geom_name=name, parent_node_name=parent)

        # The camera starts at the open side. Back and side walls, floor, and
        # ceiling form a room volume without ever hiding generated furniture.
        half_width, half_length = width / 2, length / 2
        add_box("Floor", [width, 0.05, length], [0, -0.025, -half_length + 0.15], floor_rgba, "Room_Envelope")
        add_box("Back_Wall", [width, height, 0.08], [0, height / 2, -length + 0.15], wall_rgba, "Room_Envelope")
        add_box("Left_Wall", [0.08, height, length], [-half_width, height / 2, -half_length + 0.15], wall_rgba, "Room_Envelope")
        add_box("Right_Wall", [0.08, height, length], [half_width, height / 2, -half_length + 0.15], wall_rgba, "Room_Envelope")
        add_box("Ceiling", [width, 0.05, length], [0, height, -half_length + 0.15], ceiling_rgba, "Room_Envelope")

        for index, item in enumerate(furniture_specs or []):
            label = str(item.get("label", item.get("type", "object"))).lower()
            clean = "".join(char if char.isalnum() else "_" for char in label).strip("_") or "object"
            group = f"Object_{index:02d}_{clean}"
            scene.graph.update(frame_to=group, frame_from="Objects", matrix=identity)
            center = [float(value) for value in item.get("center", [0, 0.5, -2])]
            size = [float(value) for value in item.get("size", [0.6, 0.6, 0.6])]
            if item.get("source") or item.get("type") == "detected_object":
                GLTFExporter._add_fitted_object(scene, group, clean, center, size, item.get("yaw_degrees", 0))
                continue
            asset_name, native_width = {
                "bed": ("bed", 2.20),
                "side_table": ("side_table", 0.62),
                "nightstand": ("side_table", 0.62),
                "tv": ("tv_console", 1.90),
                "chair": ("chair", 0.58),
                "window": ("window", 2.00),
                "table": ("desk", 1.45),
                "desk": ("desk", 1.45),
                "dining_table": ("desk", 1.45),
                "coffee_table": ("coffee_table", 1.18),
                "bookshelf": ("bookshelf", 1.20),
                "bookcase": ("bookshelf", 1.20),
                "book": ("book", 0.225),
                "cabinet": ("cabinet", 1.20),
                "dresser": ("cabinet", 1.20),
                "wardrobe": ("wardrobe", 1.55),
                "sofa": ("sofa", 1.95),
                "wall_shelf": ("wall_shelf", 1.05),
                "mirror": ("mirror", 0.94),
                "wall_mirror": ("mirror", 0.94),
            }.get(clean, ("", 1.0))
            if asset_name:
                # Use one source-of-truth asset library for both the curated
                # demo and photo-based placement. Only position and scale are
                # inferred from the uploaded image.
                AssetLibrary.add_instance(
                    scene,
                    asset_name,
                    group,
                    [center[0], 0.0, center[2]],
                    scale=float(np.clip(size[0] / native_width, 0.45, 1.8)),
                    parent="Objects",
                )
            else:
                GLTFExporter._add_object_assembly(add_box, group, clean, center, size)

        Path(output_glb_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_glb_path, "wb") as output:
            output.write(scene.export(file_type="glb"))
        return output_glb_path

    @staticmethod
    def _add_fitted_object(scene, group, label, center, size, yaw):
        """Fit the complete assembly in local space, then rotate and place its center."""
        local = trimesh.Scene()
        asset = {"bed": "bed", "side_table": "side_table", "table": "desk",
                 "nightstand": "side_table", "desk": "desk", "dining_table": "desk",
                 "chair": "chair", "ergonomic_chair": "chair", "office_chair": "chair", "window": "window", "floor_lamp": "floor_lamp",
                 "table_lamp": "floor_lamp", "ceiling_lamp": "ceiling_lamp",
                 "curtain": "curtain", "curtains": "curtain",
                 "rug": "rug", "plant": "plant", "coffee_table": "coffee_table",
                 "bookshelf": "bookshelf", "bookcase": "bookshelf", "book": "book",
                 "cabinet": "cabinet", "dresser": "dresser", "wardrobe": "wardrobe", "monitor":"monitor",
                 "sofa": "sofa", "wall_shelf": "wall_shelf", "mirror": "mirror",
                 "wall_mirror": "mirror"}.get(label)
        if label in AssetLibrary.ASSETS and not asset: asset=label
        if asset:
            AssetLibrary.add_instance(local, asset, "Asset", [0,0,0], parent="world")
        elif label == "pillow":
            AssetLibrary._rounded_box(local, "Pillow", [1.0, .30, .62], [0,0,0], [244,241,234,255])
        elif label in {"blanket", "duvet", "throw"}:
            from .blanket_geometry import blanket_mesh
            vertices,quads=blanket_mesh(size)
            triangles=[tri for a,b,c,d in quads for tri in ((a,b,c),(a,c,d))]
            mesh=trimesh.Trimesh(vertices=vertices,faces=triangles,process=False)
            mesh.visual.vertex_colors=[215,211,201,255]
            local.add_geometry(mesh,node_name='Blanket_Draped_Cloth')
        else:
            def box(name, extents, position, rgba, parent):
                mesh = trimesh.creation.box(extents=extents)
                mesh.apply_translation(position)
                mesh.visual.vertex_colors = rgba
                local.add_geometry(mesh, node_name=name)
            # TV is deliberately a screen, not an unrelated TV-plus-console asset.
            GLTFExporter._add_object_assembly(box, "Parts", label, [0,0,0], size)
        bounds = local.bounds
        fit = np.eye(4)
        fit[:3,:3] = np.diag(np.array([size[0],size[2],size[1]]) / np.maximum(bounds[1]-bounds[0], 1e-6))
        fit[:3,3] = -fit[:3,:3] @ bounds.mean(axis=0)
        if label in {'blanket','duvet','throw'}:
            # Already generated at physical scale; normalizing its shallow
            # bounds back to a detection height would inflate the wrinkles.
            fit=np.eye(4)
        pose = trimesh.transformations.rotation_matrix(np.radians(yaw), [0,1,0])
        pose[:3,3] = center
        scene.graph.update(frame_to=group, frame_from="Objects", matrix=pose)
        for index, node in enumerate(local.graph.nodes_geometry):
            transform, geometry = local.graph[node]
            mesh = local.geometry[geometry].copy()
            mesh.apply_transform(fit @ transform)
            clean_part = "".join(char if char.isalnum() else "_" for char in str(node)).strip("_")
            name = f"{group}_Part_{index:03d}_{clean_part}"
            scene.add_geometry(mesh, node_name=name, geom_name=name, parent_node_name=group)

    @staticmethod
    def _add_object_assembly(add_box: Any, group: str, label: str, center: List[float], size: List[float]) -> None:
        """Build a named, grounded assembly rather than a single image-aligned box."""
        cx, _, cz = center
        width, depth, height = (max(0.15, value) for value in size)

        if label == "bed":
            base_h = min(0.24, height * 0.34)
            mattress_h = min(0.30, height * 0.42)
            add_box(f"{group}_Frame", [width, base_h, depth], [cx, base_h / 2, cz], GLTFExporter.DARK, group)
            add_box(f"{group}_Mattress", [width * 0.96, mattress_h, depth * 0.94], [cx, base_h + mattress_h / 2, cz], GLTFExporter.NEUTRAL, group)
            head_h = max(0.65, height * 1.5)
            add_box(f"{group}_Headboard", [width, head_h, 0.10], [cx, head_h / 2, cz - depth / 2 + 0.05], GLTFExporter.DARK, group)
            return

        if label in {"side_table", "table", "dining_table"}:
            top_h = min(0.10, height * 0.15)
            add_box(f"{group}_Top", [width, top_h, depth], [cx, height - top_h / 2, cz], GLTFExporter.NEUTRAL, group)
            leg_w = max(0.04, min(width, depth) * 0.12)
            for leg_index, (dx, dz) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                add_box(f"{group}_Leg_{leg_index}", [leg_w, max(0.12, height - top_h), leg_w], [cx + dx * (width / 2 - leg_w / 2), (height - top_h) / 2, cz + dz * (depth / 2 - leg_w / 2)], GLTFExporter.DARK, group)
            return

        if label == "sofa":
            seat_h = min(0.38, height * 0.45)
            add_box(f"{group}_Seat", [width, seat_h, depth], [cx, seat_h / 2, cz], GLTFExporter.NEUTRAL, group)
            back_h, back_d = max(0.35, height * 0.6), min(0.20, depth * 0.28)
            add_box(f"{group}_Back", [width, back_h, back_d], [cx, seat_h + back_h / 2, cz - depth / 2 + back_d / 2], GLTFExporter.DARK, group)
            return

        if label == "chair":
            seat_h = min(0.48, height * 0.55)
            add_box(f"{group}_Seat", [width, 0.10, depth], [cx, seat_h, cz], GLTFExporter.NEUTRAL, group)
            back_h = max(0.3, height - seat_h)
            add_box(f"{group}_Back", [width, back_h, 0.09], [cx, seat_h + back_h / 2, cz - depth / 2 + 0.045], GLTFExporter.DARK, group)
            return

        # Unknown regions remain visible and volumetric, but honest: they are
        # named visual objects instead of pretending to be a known furniture type.
        add_box(f"{group}_Volume", [width, height, depth], [cx, height / 2, cz], GLTFExporter.NEUTRAL, group)
