# Vectorworks: Creating and Setting Up Design Layers (with Elevations)

> **Software**: Vectorworks 2023/2024 (Fundamentals, Architect, Landmark, or Designer)

---

## 1. Why Layers & Elevations Matter  
Design Layers hold your model geometry at real-world size.  
Each layer stores two key height values:

| Parameter | Meaning |
|-----------|---------|
| **Elevation (Z)** | Starting height of the layer, measured from the file origin *or* its Story Level |
| **Layer Wall Height (ΔZ)** | Default “ceiling” for walls/columns placed on the layer |

If you’re using **Stories**, layer elevations can be driven automatically by Story Levels.

---

## 2. Create a Design Layer

### Method A – **Organization** dialog  
1. **Tools ▸ Organization…** (*Ctrl/Cmd + Shift + O*)
2. Click on the **Design Layers** tab setting pannel.
3. Click **New…**  
4. Fill in the dialog:
  - Click **Create a New Design Layer**  
  - Type in the **Name** (such as `01-Floor` for the first floor,`02-Floor` for the second floor)
  - Press Enter or click OK to finish the creation of new design layer
5. Click **Edit…**  (you do not need to select the layer again, it has already selected the new created layer automatically, so directly click edit will be okay)
   - Setup the Elevation.  Starting height of this floor. Default for 0 for first floor, 3000 for the second floor.
   - Press Enter or click OK to finish the editing of new design layer
6. Click **OK**  or  Press Enter → layer is created and becomes active.

---

## 3. Edit Layer Elevations

1. **Tools ▸ Organization… → Design Layers**  
2. Double-click the layer (or select & **Edit…**).  
3. Adjust **Elevation (Z)** or **Layer Wall Height (ΔZ)**.  
   *Architect/Landmark*: tick **Enable Cut Plane at Layer Elevation** if you want the plan cut to follow the layer’s Z.  
4. **OK** to update.  
   Objects whose Top/Bottom Bounds are “Layer Wall Height” will move automatically.

---

## 4. Example Layer Stack

| Layer | Z Elevation | ΔZ Height |
|-------|------------:|----------:|
| `00-Slab` | –300 mm | 300 mm |
| `00-Floor` | 0 mm | 3000 mm |
| `01-Slab` | 3000 mm | 300 mm |
| `01-Floor` | 3300 mm | 3000 mm |
| `Roof` | 6300 mm | 1200 mm |

---

## 5. Best Practices

* Keep **one plan** per Design Layer—avoid stacking different floors on the same layer.  
* Prefix names with numbers (`00-`, `01-`, `02-` …) for correct ordering.  
* Turn on **Stories** early in multi-storey projects and link layers to Story Levels.  
* Save visibility presets with **Saved Views** (e.g., “Floor 1 – Architectural”).  
* Add a *Project Setup* legend on a Sheet Layer to document your elevation logic for collaborators.

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| **Elevation field is greyed out** | Layer is tied to a Story Level—edit the Story or unlink the layer. |
| **Walls ignore layer height** | In the Object Info Palette, set Top/Bottom Bounds to “Layer Wall Height” or a correct Level. |
| **Layer order looks wrong** | In **Organization ▸ Design Layers**, drag layers or use **Move Up/Down** arrows to reorder. |

---

### References  
- *Vectorworks Online Help 2024*: “Creating layers”  
- *Vectorworks Online Help 2024*: “Setting design layer properties”
