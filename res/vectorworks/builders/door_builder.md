# Vectorworks **Door Tool** — Practical Guide  
*Tested in Vectorworks 2025; the same workflow applies to 2023-2024 unless noted.*

---

## 1 Why Use the Door Tool?  
The Door tool inserts fully parametric doors that automatically cut walls, report to schedules, and support IFC data-rich BIM. It covers swing, sliding, folding, barn, pocket and curtain-wall doors, plus manufacturer catalog items.&#8203;:contentReference[oaicite:0]{index=0}  

---

## 2 Before You Start – Checklist  

| ✔ | Reason |
|----|--------|
| **Active design layer & correct elevation** | Doors inherit Z from the layer; set it first (see 2-A Layer guide). |
| **Wall selected or ready** | Doors need a host wall to cut. |
| **Resource Manager open** | Lets you drag-drop or pre-pick *Door Styles*. |
| **Data Tags loaded (optional)** | Modern schedules prefer Data Tags over legacy ID bubbles. |

---

## 3 Launching the Tool  

1. **Activate Door tool**: *Building Shell ▸ Door* (shortcut **Alt + Shift + D** Win / **⌥ + Shift + D** Mac).&#8203;:contentReference[oaicite:1]{index=1}  
2. On the Tool Bar choose:  
   * **Insertion mode** – *Standard* (centred on click) or *Offset* (sets jamb offset).&#8203;:contentReference[oaicite:2]{index=2}  
   * **Alignment** – Left | Centre | Right relative to wall thickness.  
3. Pick a **Door Style** or hit **Preferences…** to open *Door Settings* for an unstyled door.  

---

### 3-A Quick-start: Inserting Your First Door  

| Step | Action |
|------|--------|
| **1** | Activate the door tool. |
| **2** | Move cursor to the wall at the desired location. A live door preview appears with handing arrows. |
| **3** |**Double Click**, or click once then press enter to finish; the door put on the wall. |

> **Tips while inserting**  
> • **Tab** to enter exact width/offset numerically.  
> • **`J`** cycles through alignment modes on the fly.  
> • Drag the preview’s **handing indicator** to flip swing before the second click.&#8203;:contentReference[oaicite:3]{index=3}  

---

## 4 Door Settings (Dialog Cheat-Sheet)  

| Tab | What you control | Where it shows in OIP |
|-----|------------------|-----------------------|
| **General** | Width, Height, Configuration (Swing, Sliding…), Top Shape, Interior Side | Size/Config |
| **Leaf** | Panel rails, stiles, muntins | Leaf |
| **Jambs** | Jamb depth, shim gap, casings | Jamb |
| **Transom / Sidelight** | Add lights and set glazing | Lite |
| **Hardware** | Handles, hinges, threshold | Hardware |
| **2D / 3D Visualization** | Classes, line styles, open angle | Visualization |
| **ID Tag** | Prefix/Label/Suffix, bubble, include in schedule | ID Tag |
| **Data** | IFC, custom records | Data | :contentReference[oaicite:4]{index=4}  

> **Instance vs Style vs Catalog** icons in the left column tell you which fields are locked by the style or a manufacturer catalog.&#8203;:contentReference[oaicite:5]{index=5}  

---

## 5 Door Styles in Practice  

### 5-1 Create or Import  
1. **Resource Manager ▸ New Resource ▸ Door Style** (or duplicate one).  
2. Edit the tabs as above, save. The style now appears in the Resource Selector.&#8203;:contentReference[oaicite:6]{index=6}  

### 5-2 Apply / Replace  
*Before inserting:* pick the style in the Tool Bar.  
*After inserting:* select door ▸ **Replace** in Object Info Palette (OIP) or drag a style onto the door.  

### 5-3 Convert to Unstyled  
Select door ▸ OIP ▸ **Convert to Unstyled Door** to unlock every parameter for that instance.  

---

## 6 Editing Existing Doors  

| Method | Use it for | How |
|--------|------------|-----|
| **Interactive handing** | Flip swing or change interior side | Grab the handing arrow with the Selection tool and drag.&#8203;:contentReference[oaicite:7]{index=7} |
| **OIP fields** | Width, height, jamb depth, ID, replace style | Select door → edit fields. |
| **Door Settings dialog** | Any by-instance parameter | Double-click door or click **Settings** in OIP.&#8203;:contentReference[oaicite:8]{index=8} |
| **Reshape tool** | Graphically stretch rough opening in Top/Plan | Drag handles. |
| **Flip / Mirror** | Swap hinge side or copy mirrored doors | Modify menu or hotkeys. |

---

## 7 Schedules & ID Tags  

1. **Include on schedule** checkbox in the *ID Tag* pane (or Data Tag tool) ensures the door appears in *Tools ▸ Reports ▸ Create Door Schedule*.&#8203;:contentReference[oaicite:9]{index=9}  
2. Use **Data Tag** instead of legacy ID bubbles for BIM-compliant tagging; tags auto-update when doors renumber.  

---

## 8 Shortcut Reference  

| Key / Combo | Result |
|-------------|--------|
| **⌥ + Shift + D / Alt + Shift + D** | Activate Door tool |
| **`J`** | Cycle alignment (Left/Centre/Right) |
| **Tab** | Numeric input mid-insert |
| **`Shift`** | Constrain horizontal/vertical while inserting |
| **X X** | Cancel insertion |

---

## 9 Troubleshooting  

| Symptom | Fix |
|---------|-----|
| Door won’t cut the wall | Ensure it’s inserted *inside* a wall; drag slightly till preview snaps. |
| Handing arrows won’t move | That parameter is “By Style”; edit the style or convert to unstyled. |
| Width field is greyed out | Door Style locks it; set to By Instance or unstyle the door. |
| Door missing from schedule | Verify **Include on schedule** is ticked and the door’s class is visible. |
| Door not flipping with mirror | Door Symbol may be non-parametric; use parametric door or check *Flip* options in OIP. |

---

### 10 More Learning Resources  
* **Vectorworks University – “Door Tool Settings” video course** (free).&#8203;:contentReference[oaicite:10]{index=10}  
* **Help ▸ Doors ▸ Door Settings** for exhaustive parameter details.&#8203;:contentReference[oaicite:11]{index=11}  

*Happy door-placing!*  
