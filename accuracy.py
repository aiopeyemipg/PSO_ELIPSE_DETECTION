import json

def get_ellipse_properties(json_path, image_id):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Search for the selected image_id
    for annotation in data["annotations"]:
        if annotation["image_id"] == image_id:
            
            # If there are multiple ellipses, take the first one
            ellipse = annotation["ellipses"][0]
            
            return {
                "center_x": ellipse["center_x"],
                "center_y": ellipse["center_y"],
                "semi_major_axis": ellipse["semi_major_axis"],
                "semi_minor_axis": ellipse["semi_minor_axis"],
                "orientation_angle_rad": ellipse["orientation_angle_rad"]
            }

    return "Image ID not found"


# Example usage
json_file = "annotations.json"
result = get_ellipse_properties(json_file, 2)
print(result)
