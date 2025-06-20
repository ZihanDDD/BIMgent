# Vectorworks **Window Tool** — Practical Guide  
*Tested in Vectorworks 2025; the same workflow applies to 2023-2024 unless noted.*

---

## 1 Why Use the Window Tool?  
The Window tool inserts fully parametric windows that automatically cut walls, schedule themselves, and carry IFC/Energos data. It handles swing, sliding, fixed, corner, span and curtain-wall windows, plus manufacturer-catalog items. :contentReference[oaicite:0]{index=0}  

---

---

## 3 Launching the Tool  

1. **Activate Window tool**: *Building Shell ▸ Window* (shortcut **Shift + D** Win / **⌥ Shift + W** Mac). :contentReference[oaicite:1]{index=1}  
2. On the Tool Bar choose:  
   * **Insertion mode** – *Interactive Sizing* (Corner-to-Corner or Center-to-Corner).  
   * **Wall Insertion ON/OFF** icon if you need a free-standing window.  

---

### 3-A Quick-start: Inserting Your First Window  

| Step | Action |
|------|--------|
| **1** | Activate the window tool. |
| **2** | move the window to a wall,Hover over a wall until it highlights. A live preview with handing arrows appears. |
| **3** | **Double Click**, or click once then press enter to finish; the window is cut into the wall. |

> **Tip:** Hit **Tab** mid-drag to type an exact width/height or sill elevation.  

---

## 4 Window Settings — Dialog Cheat-Sheet  

| Tab | What you control |
|-----|------------------|
| **General** | Width, Height, Shape, Elevation-in-Wall, Corner/Span options |
| **Sash** | Frame thickness, unequal sashes, rails & stiles |
| **Jambs** | Jamb depth, shim gap, interior/exterior casings |
| **Lite (Glazing)** | Muntins, grids, glass class/material |
| **Transom / Sidelight** | Add glazed panels above/beside window |
| **Mullion** | For stacked or grouped units |
| **2D / 3D Visualization** | Classes, pen/fill, open angle, detail level |
| **Energos** | U-value, shading factors for energy analysis |
| **ID Tag** | Prefix/Label/Suffix, bubble shape, include on schedule |
| **Data** | IFC, custom records | :contentReference[oaicite:4]{index=4}  

> Icons in the left column show whether each parameter is **By Instance**, **By Style**, or locked to a **Catalog**. :contentReference[oaicite:5]{index=5}  

---

## 5 Window Styles in Practice  

### 5-1 Create or Import  
1. **Resource Manager ▸ New Resource ▸ Window Style** (or duplicate one).  
2. Edit settings as above, then **OK**. The style now appears in the Resource Selector. :contentReference[oaicite:6]{index=6}  

### 5-2 Apply / Replace  
*Before inserting*: choose the style in the Tool Bar.  
*After inserting*: select window ▸ **Replace** in the Object Info Palette (OIP) or drag a style onto the window. :contentReference[oaicite:7]{index=7}  

### 5-3 Convert to Unstyled  
Select window ▸ OIP ▸ **Convert to Unstyled Window** to unlock every parameter for that instance. :contentReference[oaicite:8]{index=8}  

---

## 6 Editing Existing Windows  

| Method | Use it for | How |
|--------|------------|-----|
| **Interactive handing** | Flip swing, interior side | Drag the handing arrow with Selection tool. :contentReference[oaicite:9]{index=9} |
| **OIP fields** | Size, sill/head elevation, ID, replace style | Select window → edit. |
| **Window Settings dialog** | Any by-instance parameter | Double-click window or click **Settings**. :contentReference[oaicite:10]{index=10} |
| **Reshape tool** | Graphically stretch rough opening | Drag handles in Top/Plan or 3-D. |
| **Mirror / Flip** | Copy mirrored windows | Modify menu or hotkeys. |

---

## 7 Schedules & ID Tags  

1. Tick **Include on schedule** in the *ID Tag* pane so the window appears in `Tools ▸ Reports ▸ Create Window Schedule`. :contentReference[oaicite:11]{index=11}  
2. Prefer **Data Tags** over legacy bubbles; tags update automatically when windows renumber.  

---

## 8 Shortcut Reference  

| Key / Combo | Result |
|-------------|--------|
| **Shift + D / ⌥ Shift + W** | Activate Window tool |
| **Shift** | Constrain square while sizing |
| **Ctrl Shift / ⌘ Shift** | Constrain to golden ratio |
| **Tab** | Numeric input while sizing |
| **X X** | Cancel insertion |

---

## 9 Troubleshooting  

| Symptom | Fix |
|---------|-----|
| Window won’t cut the wall | Make sure Wall Insertion mode is on and you clicked inside a wall. |
| Handing arrows don’t move | Parameter is “By Style”; edit the style or convert to unstyled. |
| Width/Height fields are greyed | Locked by style or catalog. |
| Window missing from schedule | Verify **Include on schedule** and that its class is visible. |
| Window doesn’t mirror correctly | Use parametric window, not a symbol, or check **Flip** in OIP. |

---

### 10 More Learning Resources  

* **Vectorworks University – “Window Tool Settings” video course** (free). :contentReference[oaicite:12]{index=12}  
* **Help ▸ Windows ▸ Window Settings** for exhaustive parameter descriptions. :contentReference[oaicite:13]{index=13}  

*Happy window-placing!*  
