"""
Kawempe Flood Exposure Dashboard (3-column layout)

Run:  python -m streamlit run dashboard.py

Put these files in the SAME folder as dashboard.py (names can be changed in FILES below):
  kawempe_buildings_elev.csv      (from Earth Engine: lon, lat, elev, area_m2)
  kawempe_village_stats.geojson   (from Earth Engine: village polygons + pop_sum, elev_*, bld_*)
  roads.geojson                   (from QGIS, optional)
  kawempe_rain_monthly.csv        (from Earth Engine, optional: month, rain_mm)
  kawempe_rain_annual_max.csv     (from Earth Engine, optional: year, max_daily_mm)
"""
import os

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point
from shapely.ops import unary_union
from streamlit_folium import st_folium

# ------------------------------------------------------------------ SETTINGS
DATA_DIR = "."
FILES = {
    "buildings": "kawempe_buildings_elev.csv",
    "villages": "kawempe_village_stats.geojson",
    "roads": "roads.geojson",
    "rain_monthly": "kawempe_rain_monthly.csv",
    "rain_max": "kawempe_rain_annual_max.csv",
}
VILLAGE_COL = None          # set to your village-name column, or leave None to auto-detect
RAIN_LEVELS = [50, 100, 150, 200, 250, 300]   # columns of the heatmap (mm)
UTM = 32636                 # UTM zone 36N, used for distances in metres
MAX_POINTS_ON_MAP = 1500

st.set_page_config(page_title="Kawempe Flood Exposure", layout="wide")


def fpath(key):
    return os.path.join(DATA_DIR, FILES[key])


# ------------------------------------------------------------------ LOAD DATA
def detect_village_col(gdf):
    if VILLAGE_COL and VILLAGE_COL in gdf.columns:
        return VILLAGE_COL
    cols = [c for c in gdf.columns if c != "geometry"]
    names = ("village", "villagename", "village_na", "village_nam", "vname", "v_name", "name")
    for c in cols:
        if c.lower() in names:
            return c
    for c in cols:
        if gdf[c].dtype == object and not c.lower().startswith("system"):
            return c
    return cols[0]


def keep_only(g, wanted):
    """Drop empty shapes and unpack mixed 'geometry collections' (common after QGIS clips)."""
    if g is None or g.is_empty:
        return None
    if g.geom_type == "GeometryCollection":
        parts = [p for p in g.geoms if p.geom_type in wanted]
        return unary_union(parts) if parts else None
    return g if g.geom_type in wanted else None


def clean(gdf, wanted):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: keep_only(g, wanted))
    gdf = gdf[gdf.geometry.notna()]
    return gdf[~gdf.geometry.is_empty].reset_index(drop=True)


@st.cache_data
def load_all():
    vill = gpd.read_file(fpath("villages")).to_crs(4326)
    vill = clean(vill, ("Polygon", "MultiPolygon"))
    vcol = detect_village_col(vill)
    vill = vill.rename(columns={vcol: "village"})
    vill["village"] = vill["village"].astype(str)
    vill["vid"] = vill.index
    for c in ["pop_sum", "bld_total", "bld_road", "elev_mean"]:
        if c not in vill.columns:
            vill[c] = np.nan

    b = pd.read_csv(fpath("buildings")).dropna(subset=["lon", "lat", "elev"])
    pts = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b["lon"], b["lat"]), crs=4326)
    pts = gpd.sjoin(pts, vill[["vid", "geometry"]], how="left", predicate="within")
    pts = pts.drop(columns="index_right")
    pts = pts[~pts.index.duplicated()]
    pts = pts.dropna(subset=["vid"]).reset_index(drop=True)   # only buildings inside villages
    pts["vid"] = pts["vid"].astype(int)

    roads = None
    if os.path.exists(fpath("roads")):
        roads = gpd.read_file(fpath("roads")).to_crs(4326)[["geometry"]]
        roads = clean(roads, ("LineString", "MultiLineString"))
        roads["geometry"] = roads.geometry.simplify(0.00003)
        roads = clean(roads, ("LineString", "MultiLineString"))
        if roads.empty:
            roads = None

    rm = pd.read_csv(fpath("rain_monthly")) if os.path.exists(fpath("rain_monthly")) else None
    ra = pd.read_csv(fpath("rain_max")) if os.path.exists(fpath("rain_max")) else None
    return vill, pts, roads, rm, ra, vcol


missing = [FILES[k] for k in ("buildings", "villages") if not os.path.exists(fpath(k))]
if missing:
    st.error(
        "Missing required file(s): " + ", ".join(missing)
        + ". Put them in the same folder as dashboard.py (see the top of the file)."
    )
    st.stop()

vill, pts, roads, rain_monthly, rain_max, detected_col = load_all()
elev = pts["elev"].to_numpy()

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.title("Kawempe Flood Exposure")
st.sidebar.caption("Scenario controls")

events = {}
if rain_max is not None and {"year", "max_daily_mm"} <= set(rain_max.columns):
    for _, r in rain_max.dropna().iterrows():
        events[f"{int(r['year'])}: wettest day ({r['max_daily_mm']:.0f} mm)"] = float(r["max_daily_mm"])

choice = st.sidebar.selectbox("Rainfall scenario", ["Custom slider"] + list(events.keys()))
custom_rain = st.sidebar.slider("Rainfall (mm / 24h)", 0, 300, 100, 10,
                                disabled=choice != "Custom slider")
rain = custom_rain if choice == "Custom slider" else events[choice]

runoff = st.sidebar.slider("Runoff coefficient", 0.1, 1.0, 0.6, 0.05)

lo, hi = float(np.floor(elev.min())), float(np.ceil(np.median(elev)))
p05 = float(np.clip(round(np.percentile(elev, 5) * 2) / 2, lo, hi))
base = st.sidebar.slider("Base water level (m a.s.l.)", lo, hi, p05, 0.5,
                         help="Starting level of the drainage channel / wetland. "
                              "Default = 5th percentile of building elevations.")
radius = st.sidebar.slider("Click-query radius (m)", 100, 1500, 400, 100)
show_pts = st.sidebar.checkbox("Show flooded building points", True)
min_n = st.sidebar.slider("Min. sampled buildings per village", 1, 30, 5,
                          help="Villages with fewer sampled buildings are greyed out and left "
                               "out of the ranking and heatmap, because their % is unreliable.")

# ------------------------------------------------------------------ ANALYSIS (bathtub model)
rise = rain * runoff / 25.0
level = base + rise
pts["flooded"] = pts["elev"] < level

g = pts.dropna(subset=["vid"]).groupby("vid").agg(n_sample=("flooded", "size"),
                                                  n_flood=("flooded", "sum"))
g["pct"] = 100 * g["n_flood"] / g["n_sample"]
vs = vill.merge(g, left_on="vid", right_index=True, how="left")
for c in ["n_sample", "n_flood", "pct"]:
    vs[c] = vs[c].fillna(0)
vs["est_bld"] = vs["pct"] / 100 * vs["bld_road"].fillna(vs["n_sample"])
# people = flooded buildings x average persons per building in the village
ppb = vs["pop_sum"] / vs["bld_total"].where(vs["bld_total"] > 0)
vs["pop_aff"] = (vs["est_bld"] * ppb).fillna(vs["pct"] / 100 * vs["pop_sum"].fillna(0))
vs["pop_aff"] = vs["pop_aff"].clip(upper=vs["pop_sum"].fillna(np.inf))
vs["reliable"] = vs["n_sample"] >= min_n

pct_bld = 100 * pts["flooded"].mean()
pop_total = float(vs["pop_sum"].fillna(0).sum())
pop_aff = float(vs["pop_aff"].sum())
pct_pop = 100 * pop_aff / pop_total if pop_total > 0 else 0.0


def gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "valueformat": ".0f"},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
    ))
    fig.update_layout(height=190, margin=dict(l=15, r=15, t=45, b=0))
    return fig


# ------------------------------------------------------------------ PAGE
tab_dash, tab_rain, tab_methods = st.tabs(["Dashboard", "Rainfall", "Methods & assumptions"])

with tab_dash:
    # ---- Row 1: map (left) + impact cards and gauges (right)
    map_col, side_col = st.columns([1.7, 1])

    # ---- Column 2: map + heatmap
    with map_col:
        st.subheader("Flood exposure by village")
        b = vill.total_bounds
        m = folium.Map(location=[(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], zoom_start=14,
                       tiles=None, control_scale=True)
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

        ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
        basemaps = [
            ("Light gray", ESRI.format("Canvas/World_Light_Gray_Base"), "Tiles &copy; Esri", 16, True),
            ("Streets", ESRI.format("World_Street_Map"), "Tiles &copy; Esri", 19, False),
            ("Satellite", ESRI.format("World_Imagery"), "Tiles &copy; Esri, Maxar, Earthstar Geographics", 19, False),
            ("Topographic", ESRI.format("World_Topo_Map"), "Tiles &copy; Esri", 19, False),
            ("Terrain (OpenTopoMap)", "https://tile.opentopomap.org/{z}/{x}/{y}.png",
             "Map data &copy; OpenStreetMap contributors, SRTM | Style &copy; OpenTopoMap (CC-BY-SA)", 17, False),
        ]
        for bm_name, bm_url, bm_attr, bm_zmax, bm_on in basemaps:
            folium.TileLayer(tiles=bm_url, attr=bm_attr, name=bm_name,
                             max_zoom=bm_zmax, show=bm_on).add_to(m)

        cmap = cm.linear.YlOrRd_09.scale(0, 100)
        cmap.caption = "% of sampled buildings flooded"
        show = vs[["village", "pct", "n_sample", "est_bld", "pop_aff", "geometry"]].copy()
        show["n_sample"] = show["n_sample"].astype(int)
        show["pct"] = show["pct"].round(1)
        show["est_bld"] = show["est_bld"].round(0)
        show["pop_aff"] = show["pop_aff"].round(0)
        folium.GeoJson(
            show,
            style_function=lambda f: {
                "fillColor": ("#bdbdbd" if f["properties"]["n_sample"] < min_n
                              else cmap(f["properties"]["pct"])),
                "color": "#444", "weight": 0.8, "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["village", "pct", "n_sample", "est_bld", "pop_aff"],
                aliases=["Village", "% flooded", "Sampled buildings", "Est. buildings", "Est. people"]),
            name="Villages",
        ).add_to(m)
        # compact custom legend (bottom-left) instead of the wide default one
        stops = ", ".join(cmap(v)[:7] for v in (0, 25, 50, 75, 100))
        legend = f'''
        <div style="position:fixed; bottom:24px; left:12px; z-index:9999;
                    background:rgba(255,255,255,.92); padding:8px 10px; border-radius:6px;
                    font:12px/1.4 sans-serif; color:#222; box-shadow:0 1px 5px rgba(0,0,0,.35);">
          <b>% buildings flooded</b>
          <div style="width:150px;height:10px;margin:4px 0 2px;
                      background:linear-gradient(to right,{stops});"></div>
          <div style="display:flex;justify-content:space-between;">
            <span>0</span><span>50</span><span>100</span></div>
          <div style="margin-top:4px;"><span style="display:inline-block;width:10px;height:10px;
                      background:#bdbdbd;"></span> under {min_n} sampled buildings (not rated)</div>
          <div><span style="color:#1971c2;">&#9679;</span> flooded building &nbsp;
               <span style="color:#e8590c;">&#9644;</span> road</div>
        </div>'''
        m.get_root().html.add_child(folium.Element(legend))

        if roads is not None:
            folium.GeoJson(
                roads,
                style_function=lambda f: {"color": "#e8590c", "weight": 2.2, "opacity": 0.9},
                name="Roads",
            ).add_to(m)
        if show_pts:
            fg = folium.FeatureGroup(name="Flooded buildings")
            for _, r in pts[pts["flooded"]].head(MAX_POINTS_ON_MAP).iterrows():
                folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3, color="white",
                                    weight=0.6, fill=True, fill_color="#1971c2",
                                    fill_opacity=0.95).add_to(fg)
            fg.add_to(m)
        folium.LayerControl(collapsed=True, position="topright").add_to(m)

        out = st_folium(m, height=480, use_container_width=True, returned_objects=["last_clicked"])

    with side_col:
        st.subheader("Flood impact")
        st.metric("Flood water level", f"{level:.1f} m", f"+{rise:.1f} m rise")
        st.metric("Est. buildings flooded", f"{int(round(vs['est_bld'].sum())):,}")
        st.metric("Est. people affected", f"{int(round(pop_aff)):,}")
        g1, g2 = st.columns(2)
        g1.plotly_chart(gauge(pct_bld, "Buildings flooded", "#d62728"))
        g2.plotly_chart(gauge(pct_pop, "Population affected", "#ff7f0e"))

    # ---- Row 2: heatmap (left) + ranking, query and about (right)
    row2_left, row2_right = st.columns([1.7, 1])

    with row2_left:

        # ---- heatmap: village x rainfall level
        st.subheader("Flooded % by village and rainfall level")
        top = vs[vs["reliable"]].sort_values("pop_aff", ascending=False).head(15)
        levels = [base + r * runoff / 25.0 for r in RAIN_LEVELS]
        rows, labels = [], []
        for _, v in top.iterrows():
            e = pts.loc[pts["vid"] == v["vid"], "elev"].to_numpy()
            rows.append([100 * (e < lv).mean() if len(e) else 0.0 for lv in levels])
            labels.append(v["village"])
        if rows:
            hm = px.imshow(pd.DataFrame(rows, index=labels, columns=[str(r) for r in RAIN_LEVELS]),
                           color_continuous_scale="YlOrRd", zmin=0, zmax=100, aspect="auto",
                           labels={"x": "Rainfall (mm / 24h)", "y": "", "color": "% flooded"})
            hm.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(hm)
            st.caption("Top 15 villages by people affected in the current scenario.")

    # ---- Column 3: ranked table + query + about
    with row2_right:
        st.subheader("Most exposed villages")
        table = (vs[vs["reliable"]].sort_values("pct", ascending=False).head(10)
                 [["village", "pct", "pop_aff"]])
        table = table.rename(columns={"village": "Village", "pct": "% flooded",
                                      "pop_aff": "People affected"})
        table["People affected"] = table["People affected"].round(0).astype(int)
        st.dataframe(
            table, hide_index=True,
            column_config={"% flooded": st.column_config.ProgressColumn(
                "% flooded", min_value=0, max_value=100, format="%.0f%%")},
        )

        st.subheader("Spatial query")
        click = out.get("last_clicked") if out else None
        if click:
            pt = gpd.GeoSeries([Point(click["lng"], click["lat"])], crs=4326).to_crs(UTM)
            buf = pt.buffer(radius).iloc[0]
            near = pts[pts.to_crs(UTM).within(buf)]
            inside = vill[vill.contains(Point(click["lng"], click["lat"]))]
            if len(inside):
                vname = inside.iloc[0]["village"]
                st.write(f"**Village:** {vname}")
            st.write(f"Within **{radius} m** of your click:")
            st.write(f"- Sampled buildings: **{len(near)}**")
            st.write(f"- Flooded: **{int(near['flooded'].sum())}**")
            if len(near):
                st.write(f"- Mean elevation: **{near['elev'].mean():.1f} m**")
        else:
            st.info("Click the map to query nearby buildings.")

        with st.expander("About"):
            st.markdown(
                "- **Buildings:** Google Open Buildings v3 (sample along roads)\n"
                "- **Elevation:** SRTM 30 m\n"
                "- **Population:** WorldPop 2020\n"
                "- **Rainfall:** CHIRPS\n"
                "- **Roads:** OpenStreetMap\n\n"
                "Full details are in the **Methods & assumptions** tab.")

with tab_rain:
    st.subheader("Rainfall context (CHIRPS)")
    if rain_monthly is not None and {"month", "rain_mm"} <= set(rain_monthly.columns):
        st.plotly_chart(px.line(rain_monthly, x="month", y="rain_mm",
                                labels={"month": "Month", "rain_mm": "Rainfall (mm / month)"},
                                title="Monthly rainfall"))
    if rain_max is not None and {"year", "max_daily_mm"} <= set(rain_max.columns):
        st.plotly_chart(px.bar(rain_max, x="year", y="max_daily_mm",
                               labels={"year": "Year", "max_daily_mm": "Max daily rainfall (mm)"},
                               title="Wettest day each year"))
        st.caption("Pick one of these years in the sidebar to run that event as a scenario.")
    if rain_monthly is None and rain_max is None:
        st.info("No rainfall files found. Add the CHIRPS exports to see charts here.")

with tab_methods:
    st.markdown(f"""
### Study area
Kawempe Division, Kampala (Uganda), analysed at village level.

### Data sources
- **Village boundaries:** *(add source, e.g. KCCA / UBOS / HDX)*
- **Buildings:** Google Open Buildings v3 (confidence >= 0.75, area >= 30 m2), kept if within
  150 m of the road layer; random sample of up to 8,000 points
- **Elevation:** SRTM 30 m (USGS/SRTMGL1_003), sampled at building centroids
- **Population:** WorldPop 100 m, 2020 (village totals)
- **Rainfall:** CHIRPS daily (about 5 km), 2015-2025
- **Roads:** OpenStreetMap highways, major classes *(state the classes you kept; add download date)*

### Current scenario parameters
- Rainfall: {rain:.0f} mm / 24h; runoff coefficient: {runoff}
- Water rise = rainfall x runoff / 25 = {rise:.1f} m
- Base water level: {base} m, flood level: {level:.1f} m

### Model
Bathtub model: a building is flooded if its ground elevation is below the flood level.

### Assumptions
- Uniform rainfall and runoff over the whole division
- No flow routing, drainage capacity, infiltration or building floor height
- Only buildings inside the village boundaries and within 150 m of the roads are analysed, so results
  describe road-side exposure, not whole villages
- Sampled buildings represent all buildings along roads in their village
- Estimated buildings flooded = share of sampled buildings flooded x buildings along roads in the village
- People affected = estimated flooded buildings x average persons per building in the village
  (WorldPop village total / all buildings in the village, assumed uniform)
- Villages with fewer sampled buildings than the chosen minimum ({min_n}) are greyed out and left out
  of the ranking and heatmap

### Limitations
- SRTM 30 m is coarse for flat urban terrain; building elevations are approximate
- CHIRPS is too coarse to show neighbourhood-scale rainfall differences
- Indicative results only; this is not a hydrodynamic model
- Village column used: `{detected_col}`
""")
