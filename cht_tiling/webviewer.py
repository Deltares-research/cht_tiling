def write_html(
    file_name, title="CHT Tiling", legend_title="Legend", max_native_zoom=19
):
    # add carriage return to each line of the file

    # open html file for writing
    f = open(file_name, "w")
    f.write("<!DOCTYPE html>\r\n")
    f.write("<head>\r\n")
    f.write(
        "  <meta http-equiv='content-type' content='text/html; charset=UTF-8' />\r\n"
    )
    f.write("  <script>\r\n")
    f.write("    L_NO_TOUCH = false;\r\n")
    f.write("    L_DISABLE_3D = false;\r\n")
    f.write("  </script>\r\n")
    f.write(
        "  <style>html, body {width: 100%;height: 100%;margin: 0;padding: 0;}</style>\r\n"
    )
    f.write(
        "  <script src='https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.js'></script>\r\n"
    )
    f.write(
        "  <script src='https://code.jquery.com/jquery-1.12.4.min.js'></script>\r\n"
    )
    f.write(
        "  <script src='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js'></script>\r\n"
    )
    f.write(
        "  <script src='https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js'></script>\r\n"
    )
    f.write(
        "  <link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/leaflet@1.6.0/dist/leaflet.css'/>\r\n"
    )
    f.write(
        "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css'/>\r\n"
    )
    f.write(
        "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css'/>\r\n"
    )
    f.write(
        "  <link rel='stylesheet' href='https://maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css'/>\r\n"
    )
    f.write(
        "  <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css'/>\r\n"
    )
    f.write(
        "  <link rel='stylesheet' href='https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css'/>\r\n"
    )
    f.write(
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no' />\r\n"
    )
    # f.write("  <script src='leaflet-velocity.js'></script>\r\n")
    f.write("\r\n")
    f.write("  <style>\r\n")
    f.write("      #map { width: 800px; height: 500px; }\r\n")
    f.write(
        "      .info { padding: 6px 8px; font: 17px/19px Arial, Helvetica, sans-serif; background: white; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; } .info h4 { margin: 0 0 5px; color: #777; }\r\n"
    )
    f.write(
        "      .legend     { text-align: center; line-height: 18px; color: #555; } .legend i     { width: 20px; height: 15px; float: left; margin-right: 8px; opacity: 0.7; border-style: solid; border-width: 1px;}\r\n"
    )
    # f.write("      .wind { padding: 6px 8px; font: 17px/19px Arial, Helvetica, sans-serif; background: white; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; } .info h4 { margin: 0 0 5px; color: #777; }\r\n")
    # f.write("      .windLegend { text-align: center; line-height: 18px; color: #555; } .windLegend i { width: 50px; height: 18px; float: left; margin-right: 8px; opacity: 0.7; }\r\n")
    f.write("  </style>\r\n")
    f.write("\r\n")
    f.write("</head>\r\n")
    f.write("<body>\r\n")
    f.write("  <h3> " + title + "</h3>\r\n")
    f.write("  <div id='map' style='width: 100%; height: 90%;'></div>\r\n")
    f.write("</body>\r\n")
    f.write("<script>\r\n")
    f.write("\r\n")
    f.write("// Base layers\r\n")
    f.write(
        "var tile_layer_osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',\r\n"
    )
    f.write(
        "    {'attribution': 'Data by http://openstreetmap.org href=http://www.openstreetmap.org',\r\n"
    )
    f.write("     'detectRetina': false,\r\n")
    f.write("     'maxZoom': 19,\r\n")
    f.write("     'minZoom': 0,\r\n")
    f.write("     'noWrap': false,\r\n")
    f.write("     'opacity': 1,\r\n")
    f.write("     'maxNativeZoom': 13,\r\n")
    f.write("     'subdomains': 'abc',\r\n")
    f.write("     'tms': false});\r\n")
    f.write(
        "var Esri_WorldImagery = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {\r\n"
    )
    f.write(
        "	attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'\r\n"
    )
    f.write("});\r\n")
    f.write("\r\n")
    f.write("// Data layer\r\n")
    f.write("var tile_layer = L.tileLayer(\r\n")
    f.write("    '{z}/{x}/{y}.png',\r\n")
    f.write("    {'attribution': 'cosmos',\r\n")
    f.write("     'detectRetina': false,\r\n")
    f.write("     'opacity': 0.7,\r\n")
    f.write("     'maxNativeZoom': " + str(max_native_zoom) + ",\r\n")
    f.write("     'maxZoom': 19,\r\n")
    f.write("     'minZoom': 0,\r\n")
    f.write("     'noWrap': false,\r\n")
    f.write("     'subdomains': 'abc',\r\n")
    f.write("     'zIndex':10,\r\n")
    f.write("     'tms': false}\r\n")
    f.write(");\r\n")
    f.write("\r\n")
    f.write("var legend = L.control({position: 'bottomright'});\r\n")
    f.write("legend.onAdd = function (map) {\r\n")
    f.write("        var div = L.DomUtil.create('div', 'info legend')\r\n")
    f.write("        div.innerHTML += '" + legend_title + "<br>'\r\n")
    # f.write("        div.innerHTML += '<i style='background:#ff0000'></i>> 2 m<br>'\r\n")
    # f.write("        div.innerHTML += '<i style='background:#ffa500'></i>1 - 2 ,<br>'\r\n")
    # f.write("        div.innerHTML += '<i style='background:#ffff00'></i>0.3 - 1 m<br>'\r\n")
    # f.write("        div.innerHTML += '<i style='background:#00ff00'></i>0.05 - 0.3 m<br>'\r\n")
    f.write("        return div;\r\n")
    f.write("};\r\n")
    f.write("\r\n")
    f.write("// Map\r\n")
    f.write("var map = L.map('map',{\r\n")
    f.write("    center: [0, 0],\r\n")
    f.write("    crs: L.CRS.EPSG3857,\r\n")
    f.write("    zoom: 2,\r\n")
    f.write("    zoomControl: true,\r\n")
    f.write("    preferCanvas: false,\r\n")
    f.write("    layers: [tile_layer_osm, tile_layer]\r\n")
    f.write("    }\r\n")
    f.write(");\r\n")
    f.write("\r\n")
    f.write("legend.addTo(map);\r\n")
    f.write("\r\n")
    f.write("// Layer control\r\n")
    f.write("var baseMaps = {\r\n")
    f.write("    'Open Street Map': tile_layer_osm,\r\n")
    f.write("    'Satellite': Esri_WorldImagery\r\n")
    f.write("};\r\n")
    f.write("\r\n")
    f.write("var overlayMaps = {};\r\n")
    f.write("\r\n")
    f.write("L.control.layers(baseMaps, overlayMaps).addTo(map);\r\n")
    f.write("\r\n")
    f.write("</script>\r\n")

    f.close()
