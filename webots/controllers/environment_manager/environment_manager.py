from controller import Supervisor
from pathlib import Path

FOOT_OFFSETS = {
    "front_left": (0.36, 0.18),
    "front_right": (0.36, -0.18),
    "rear_left": (-0.30, 0.18),
    "rear_right": (-0.30, -0.18),
}

SAND_PATCH_DEF = "SAND_PATCH"
FOOT_TRAP_DEF = "FOOT_TRAP"
FOOT_VINE_DEF = "FOOT_VINE"
WORLD_INFO_DEF = "WORLD_INFO"
PATCH_HIDDEN_TRANSLATION = [0.0, 0.0, -10.0]
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scenario.ini"


def to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_color(value):
    if not value:
        return None
    parts = [p.strip() for p in value.split(',') if p.strip()]
    if len(parts) != 3:
        return None
    try:
        return [max(0.0, min(1.0, float(p))) for p in parts]
    except ValueError:
        return None


def load_config():
    config = {}
    sand = {}
    trap = {}
    vine = {}
    if not CONFIG_PATH.exists():
        return config, sand, trap, vine

    with CONFIG_PATH.open() as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if key.startswith('sand.'):
                sand[key.split('.', 1)[1]] = value
            elif key.startswith('trap.'):
                trap[key.split('.', 1)[1]] = value
            elif key.startswith('vine.'):
                vine[key.split('.', 1)[1]] = value
            else:
                config[key] = value
    return config, sand, trap, vine


def set_vec3(field, vector, label):
    if field is None:
        print(f"[environment] warning: field '{label}' is missing")
        return
    try:
        x, y, z = vector
        field.setSFVec3f([float(x), float(y), float(z)])
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"[environment] warning: could not set {label}: {exc}")


def set_float(field, value, label):
    if field is None:
        print(f"[environment] warning: field '{label}' is missing")
        return
    try:
        field.setSFFloat(float(value))
    except (TypeError, ValueError, RuntimeError) as exc:
        print(f"[environment] warning: could not set {label}: {exc}")


def update_contact_properties(supervisor, material, friction, bounce):
    world_info = supervisor.getFromDef(WORLD_INFO_DEF)
    if world_info is None:
        return
    contact_field = world_info.getField("contactProperties")
    if contact_field is None:
        return
    for index in range(contact_field.getCount()):
        node = contact_field.getMFNode(index)
        if node is None:
            continue
        mat1_field = node.getField("material1")
        mat2_field = node.getField("material2")
        if mat1_field is None or mat2_field is None:
            continue
        if material not in {mat1_field.getSFString(), mat2_field.getSFString()}:
            continue
        set_float(node.getField("coulombFriction"), friction, f"coulombFriction ({material})")
        set_float(node.getField("bounce"), bounce, f"bounce ({material})")


def hide_node(supervisor, def_name):
    node = supervisor.getFromDef(def_name)
    if node is None:
        return
    translation_field = node.getField("translation")
    if translation_field:
        set_vec3(translation_field, PATCH_HIDDEN_TRANSLATION, f"hide {def_name}")


def apply_sand_patch(supervisor, config, sand):
    sand_patch = supervisor.getFromDef(SAND_PATCH_DEF)
    if sand_patch is None:
        print("[environment] warning: sand patch not found")
        return

    foot = config.get("buriedFoot", "front_left")
    offset = FOOT_OFFSETS.get(foot)
    if offset is None:
        print(f"[environment] warning: unknown foot '{foot}'. Sand hidden")
        hide_node(supervisor, SAND_PATCH_DEF)
        return

    radius_field = sand_patch.getField("radius")
    height_field = sand_patch.getField("height")
    color_field = sand_patch.getField("color")
    material_field = sand_patch.getField("contactMaterial")

    radius = to_float(sand.get("radius"), radius_field.getSFFloat() if radius_field else 0.2)
    height = to_float(sand.get("height"), height_field.getSFFloat() if height_field else 0.12)
    top_level = to_float(config.get("topLevel"), 0.0)
    color = parse_color(sand.get("color"))
    material = config.get("material", material_field.getSFString() if material_field else "sand")
    friction = to_float(config.get("friction"), 1.8)
    bounce = to_float(config.get("bounce"), 0.0)

    translation_field = sand_patch.getField("translation")
    if translation_field:
        center_z = top_level - height * 0.5
        set_vec3(translation_field, [offset[0], offset[1], center_z], "sand translation")
    if radius_field:
        set_float(radius_field, radius, "sand radius")
    if height_field:
        set_float(height_field, height, "sand height")
    if color_field and color:
        try:
            color_field.setSFColor(color)
        except RuntimeError as exc:
            print(f"[environment] warning: could not set sand color: {exc}")
    if material_field and material:
        material_field.setSFString(material)

    update_contact_properties(supervisor, material, friction, bounce)

    print("[environment] Sand burial active")
    print(f"               foot          : {foot}")
    print(f"               radius (m)    : {radius:.3f}")
    print(f"               height (m)    : {height:.3f}")
    print(f"               top level (m) : {top_level:.3f}")
    print(f"               friction      : {friction:.3f}")


def apply_foot_trap(supervisor, config, trap):
    trap_node = supervisor.getFromDef(FOOT_TRAP_DEF)
    if trap_node is None:
        print("[environment] warning: foot trap not found")
        return

    foot = config.get("buriedFoot", "front_left")
    offset = FOOT_OFFSETS.get(foot)
    if offset is None:
        print(f"[environment] warning: unknown foot '{foot}'. Trap hidden")
        hide_node(supervisor, FOOT_TRAP_DEF)
        return

    offset_x = to_float(trap.get("offsetX"), 0.0)
    offset_y = to_float(trap.get("offsetY"), 0.0)
    offset_z = to_float(trap.get("offsetZ"), 0.0)
    friction = to_float(trap.get("friction"), 2.5)
    bounce = to_float(trap.get("bounce"), 0.0)
    material = trap.get("material", "trap")

    translation_field = trap_node.getField("translation")
    if translation_field:
        set_vec3(translation_field, [offset[0] + offset_x, offset[1] + offset_y, offset_z], "trap translation")

    material_field = trap_node.getField("contactMaterial")
    if material_field and material:
        material_field.setSFString(material)

    update_contact_properties(supervisor, material, friction, bounce)

    print("[environment] Foot trap active")
    print(f"               foot          : {foot}")
    print(f"               translation   : {[offset[0] + offset_x, offset[1] + offset_y, offset_z]}")
    print(f"               friction      : {friction:.3f}")


def apply_foot_vine(supervisor, config, vine):
    vine_node = supervisor.getFromDef(FOOT_VINE_DEF)
    if vine_node is None:
        print("[environment] warning: foot vine not found")
        return

    foot = config.get("buriedFoot", "front_left")
    offset = FOOT_OFFSETS.get(foot)
    if offset is None:
        print(f"[environment] warning: unknown foot '{foot}'. Vines hidden")
        hide_node(supervisor, FOOT_VINE_DEF)
        return

    offset_x = to_float(vine.get("offsetX"), 0.0)
    offset_y = to_float(vine.get("offsetY"), 0.0)
    offset_z = to_float(vine.get("offsetZ"), 0.05)
    rotation = to_float(vine.get("rotation"), 0.0)
    friction = to_float(vine.get("friction"), 2.0)
    bounce = to_float(vine.get("bounce"), 0.0)
    material = vine.get("material", "vine")

    translation_field = vine_node.getField("translation")
    if translation_field:
        set_vec3(translation_field, [offset[0] + offset_x, offset[1] + offset_y, offset_z], "vine translation")

    rotation_field = vine_node.getField("rotation")
    if rotation_field:
        try:
            rotation_field.setSFRotation([0, 0, 1, rotation])
        except RuntimeError as exc:
            print(f"[environment] warning: could not set vine rotation: {exc}")

    material_field = vine_node.getField("contactMaterial")
    if material_field and material:
        material_field.setSFString(material)

    update_contact_properties(supervisor, material, friction, bounce)

    print("[environment] Foot vine active")
    print(f"               foot          : {foot}")
    print(f"               translation   : {[offset[0] + offset_x, offset[1] + offset_y, offset_z]}")
    print(f"               rotation (rad): {rotation:.3f}")
    print(f"               friction      : {friction:.3f}")


def main():
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())

    config, sand_values, trap_values, vine_values = load_config()
    scenario = config.get("scenario", "none").strip().lower()

    if scenario == "sand_burial":
        apply_sand_patch(supervisor, config, sand_values)
        hide_node(supervisor, FOOT_TRAP_DEF)
        hide_node(supervisor, FOOT_VINE_DEF)
    elif scenario == "foot_trap":
        hide_node(supervisor, SAND_PATCH_DEF)
        hide_node(supervisor, FOOT_VINE_DEF)
        apply_foot_trap(supervisor, config, trap_values)
    elif scenario == "foot_vine":
        hide_node(supervisor, SAND_PATCH_DEF)
        hide_node(supervisor, FOOT_TRAP_DEF)
        apply_foot_vine(supervisor, config, vine_values)
    else:
        hide_node(supervisor, SAND_PATCH_DEF)
        hide_node(supervisor, FOOT_TRAP_DEF)
        hide_node(supervisor, FOOT_VINE_DEF)
        print(f"[environment] scenario '{scenario}' not recognised; all environment hazards hidden")

    while supervisor.step(time_step) != -1:
        pass


if __name__ == "__main__":
    main()
