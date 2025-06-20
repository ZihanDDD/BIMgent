import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import pandas as pd

#print(ifcopenshell.version)




def extract_bim_info(file_path):
    
    model = ifcopenshell.file()

    ifc_file = ifcopenshell.open(file_path)

    settings = ifcopenshell.geom.settings()
    data = []

    # ---- Walls ----
    walls = ifc_file.by_type('IfcWall')
    for idx, wall in enumerate(walls, start=1):
        shape = ifcopenshell.geom.create_shape(settings, wall)
        x = round(ifcopenshell.util.shape.get_x(shape.geometry), 2)
        y = round(ifcopenshell.util.shape.get_y(shape.geometry), 2)
        z = round(ifcopenshell.util.shape.get_z(shape.geometry), 2)
        data.append({"Type": "Wall", "Index": idx, "x": x, "y": y, "z": z})

    # ---- Slabs ----
    slabs = ifc_file.by_type('IfcSlab')
    for idx, slab in enumerate(slabs, start=1):
        shape = ifcopenshell.geom.create_shape(settings, slab)
        x = round(ifcopenshell.util.shape.get_x(shape.geometry), 2)
        y = round(ifcopenshell.util.shape.get_y(shape.geometry), 2)
        z = round(ifcopenshell.util.shape.get_z(shape.geometry), 2)
        data.append({"Type": "Slab", "Index": idx, "x": x, "y": y, "z": z})

    # ---- Windows ----
    windows = ifc_file.by_type('IfcWindow')
    for idx, window in enumerate(windows, start=1):
        shape = ifcopenshell.geom.create_shape(settings, window)
        x = round(ifcopenshell.util.shape.get_x(shape.geometry), 2)
        y = round(ifcopenshell.util.shape.get_y(shape.geometry), 2)
        z = round(ifcopenshell.util.shape.get_z(shape.geometry), 2)
        data.append({"Type": "Window", "Index": idx, "x": x, "y": y, "z": z})

    # ---- Doors ----
    doors = ifc_file.by_type('IfcDoor')
    for idx, door in enumerate(doors, start=1):
        shape = ifcopenshell.geom.create_shape(settings, door)
        x = round(ifcopenshell.util.shape.get_x(shape.geometry), 2)
        y = round(ifcopenshell.util.shape.get_y(shape.geometry), 2)
        z = round(ifcopenshell.util.shape.get_z(shape.geometry), 2)
        data.append({"Type": "Door", "length": x})

    # ---- Create DataFrame ----
    df = pd.DataFrame(data)
    df = df.to_string(index=False)

    return df


# df = extract_bim_info(ifc_file)
# print(df)