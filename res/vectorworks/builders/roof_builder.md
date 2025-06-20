# Vectorworks **Roof Tool / Create Roof** — Practical Guide  
*Tested in Vectorworks 2025; all steps work in 2023-2024 unless noted.*

---

---

### 2.1 Activate the Tool  
- Press **CTRL + Alt + Shift + 1** *(your custom shortcut)*

## 3 Launching the Tool  

### 3-A Quick-start: Your First Roof in Seven Clicks  

| # | Action |
|---|--------|
| **1** | Select all components by using select all. |
| **2** |  Activate the Tool Press **CTRL + Alt + Shift + 1** *(your custom shortcut)*
| **3** | **Pitch**: type 30°  **Bearing Inset**: 0  **Thickness**: 300 mm. |
| **4** | Click **OK** or press enter — roof appears. |
| **5** | **Double-click** roof → **Edit Roof Settings**; change one edge to **Gable** to form a gable-hip mix. |
| **6** | Select roof ▸ Object Info Palette ▸ **Insert Dormer…** to pop a quick dormer. :contentReference[oaicite:3]{index=3} |

---

## 4 Roof Settings — Dialog Cheat-Sheet  

| Tab | What you control | Key items |
|-----|------------------|-----------|
| **General** | Global pitch, bearing height, thickness, overhang, component insertion | Pitch, Bearing Inset, Thickness |
| **Edges** | Edge-by-edge shape (Hip, Gable, Clip, Half-Hip), fascia depth | Edit each edge row |
| **Components** | Materials / layers of the roof assembly | Add sheathing, insulation, finish :contentReference[oaicite:4]{index=4} |
| **Framing** | Automatic rafter, ridge, hip, valley members | Member sizes, spacing |
| **Dormers/Skylights** | Auto-dormer & skylight size, location | Qty, offset, class |
| **Attribute** | 2-D pens/fills (grayed when style-controlled) | |
| **Data** | IFC export, Energos fields | U-value, Lambda |

> A **lock** icon means the parameter is “By Style”; to edit it for this roof only, convert to **Unstyled Roof**. :contentReference[oaicite:5]{index=5}  

---

## 5 Roof Styles in Practice  

### 5-1 Create or Import  
1. Make an unstyled roof.  
2. **Right-click ▸ New Roof Style from Unstyled Roof**. :contentReference[oaicite:6]{index=6}  
3. Name, choose component classes, save. The style lives in your file/library.

### 5-2 Apply / Replace  
*Before creation:* pick the style in **Roof Settings**.  
*After creation:* select roof ▸ **Replace** in Object Info Palette or drag-drop a style. :contentReference[oaicite:7]{index=7}  

### 5-3 Convert to Unstyled  
Select roof ▸ OIP ▸ **Convert to Unstyled Roof** to unlock every parameter instance-by-instance.

---

## 6 Editing Existing Roofs  

| Method | Use it for | How |
|--------|------------|-----|
| **Edit Roof Settings** | Change pitch, edge types, add dormers | Double-click roof or OIP **Settings…**. :contentReference[oaicite:8]{index=8} |
| **Reshape tool** | Stretch or rotate the whole roof | Drag corner handles. |
| **Roof Face command** | Irregular wings/porches | Draw a 2-D shape ▸ AEC ▸ **Create Roof Face**. Roof faces join at hips/valleys. :contentReference[oaicite:9]{index=9} |
| **Edge handles** | Interactive change from Hip ↔ Gable ↔ Half-Hip | Drag a blue edge handle, choose edge type. |
| **Dormer/Skylight editing** | Move, resize, remove | Click dormer handle ▸ **Edit Roof Element**. :contentReference[oaicite:10]{index=10} |

---

## 7 Schedules & Framing  

* Roof components and thicknesses list automatically in **Tools ▸ Reports ▸ Create Roof Schedule**.  
* Tick **Include Framing Members** in **Framing tab** to auto-generate rafters and ridges for quantity take-off. :contentReference[oaicite:11]{index=11}  

---

## 8 Shortcut Reference  

| Key / Menu | Result |
|------------|--------|
| **AEC ▸ Create Roof…** | Build a parametric roof |
| **AEC ▸ Roof Face** | Make a single face from any 2-D shape |
| **Double-click** roof | Edit Roof Settings |
| **Ctrl / ⌘ E** | Reshape tool after selection |
| **Alt-drag edge handle** | Change edge to Gable/Hip interactively |

---

## 9 Troubleshooting  

| Symptom | Fix |
|---------|-----|
| Edge won’t switch to Gable | Overhang = 0 mm; increase overhang or bearing inset. |
| Roof faces mis-align | Use **Connect/Combine Roof Faces** or make faces share same bearing height. |
| Components invisible in plan | Detail level too low; increase in Viewport or Model Setup. |

---

### 10 More Learning Resources  

* **Vectorworks University – “Roof Styles” course.** :contentReference[oaicite:12]{index=12}  
* **Help ▸ Roofs ▸ Creating Roof Objects / Editing Roof Objects** for exhaustive parameter descriptions. :contentReference[oaicite:13]{index=13}  

*Raise the roof!*  
