# Vectorworks — Creating & Editing Slabs  
*(Shortcut: **Alt + Shift + 2** for the Slab tool, as you’ve assigned)*

---

## 1 What Is a Slab?
A **Slab** is Vectorworks’ BIM object for floors, ceilings, and flat roofs.  
It can be:

| Type | Typical purpose | Key feature |
|------|-----------------|-------------|
| **Free-standing slab** | Terrace, mezzanine, podium | Drawn manually in any shape |
| **Automatically bounded slab** | Story floor within surrounding walls | Slab edges stay *associated* to walls — walls move, slab updates |
| **Hybrid** (auto + manual edges) | Complex shapes (e.g., balcony notch) | Combine wall-bound and editable edges | :contentReference[oaicite:0]{index=0}

Each slab supports **components** (concrete, insulation, finish, etc.), drainage slopes, and IFC data.

---

## 2 Creating a Slab

### 2.1 Activate the Tool  
- Press **Alt + Shift + 2** *(your custom shortcut)*, **or**  
- *Tool Sets ▸ Building Shell ▸ Slab*.

In the **Tool bar**: pick a **Slab Style** or click **Preferences…** to set thickness, components, and default top/bottom bounds. :contentReference[oaicite:1]{index=1}

---

### 2.1 Workflows


#### A • Slab from Walls (automatic)  
1. Select **Picked Walls** or **Inner Boundary**.  
2. If Picked Walls: every external wall need to be clicked, each wall only need to be **clicked ONCE** (order doesn’t matter) and press **Enter**.  
   If Inner Boundary: click once inside the room outline.  
3. Slab is created and *associated* to those walls; moving a wall reshapes the slab. :contentReference[oaicite:3]{index=3}

---

## 3 Editing Slabs

| Edit task | How |
|-----------|-----|
| **Boundary / manual edges** | Right-click slab → **Edit Boundary** (Path mode). Reshape vertices like a polyline. :contentReference[oaicite:4]{index=4} |
| **Components** (layers) | OIP ▸ **Components…** to add concrete, topping, insulation, finish. | :contentReference[oaicite:5]{index=5} |
| **Height / Level** | In OIP set **Top Bound** & **Bottom Bound** to *Story Levels* or *Layer Wall Height*. |
| **Drainage slope** | OIP ▸ **Drainage Settings…**; pick drains & slope. |
| **Convert to Style / Replace Style** | OIP ▸ **Replace Slab Style…** to swap assemblies without redrawing. |

---

## 4 Best Practices

* Bind slab **components to wall components** (core to core) for clean sections.  
* Keep structural and finish slabs separate—place each on its own Design Layer (e.g., `00-Struct-Slab`, `00-Finish-Slab`).  
* Use **Story Levels** so slab elevations auto-update if floor-to-floor heights change.  
* Turn on **Clip Cube** in 3D to inspect component joins quickly.  
* Name styles with thickness & spec (`SLB-200 RC + 50 Topping`) for quick selection. :contentReference[oaicite:6]{index=6}

---

## 5 Troubleshooting

| Issue | Fix |
|-------|-----|
| Cannot pick walls (cursor shows ⃠) | Walls must be on **visible, editable** Design Layers and form a closed loop. |
| Edge won’t move | It’s an **auto-bounded** edge; convert to manual: right-click edge → **Convert to Manual Edge**. |
| Slab ignores level change | Top/Bottom Bounds set to **Layer Elevation**; switch to *Story Level* in OIP. |
| Drainage slope fails | Ensure slab has at least two drainage points and sufficient thickness. :contentReference[oaicite:7]{index=7} |
| Thickness field greyed | Slab is **Styled**; choose **Convert to Unstyled** or edit the Style resource. |

---

### References  
- *Vectorworks Online Help 2024*: “Creating slabs”, “Automatically bounded slabs”, “Editing slab geometry”, “Slab preferences”, “Creating slab components”.
