import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from ultralytics import YOLO


@dataclass
class BoundingBox:
    x1: float  # top-left x
    y1: float  # top-left y
    x2: float  # bottom-right x
    y2: float  # bottom-right y
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

@dataclass
class SceneObject:
    object_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    attributes: Dict[str, str] = None

class SceneGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.objects: Dict[int, SceneObject] = {}
        
    def add_object(self, obj: SceneObject):
        """Add an object node to the scene graph."""
        self.objects[obj.object_id] = obj
        self.graph.add_node(obj.object_id, 
                          class_name=obj.class_name,
                          bbox=obj.bbox,
                          attributes=obj.attributes)
    
    def add_relationship(self, subject_id: int, object_id: int, relationship: str):
        """Add a relationship edge between two objects."""
        if subject_id in self.objects and object_id in self.objects:
            self.graph.add_edge(subject_id, object_id, relationship=relationship)
    
    def detect_spatial_relationships(self, distance_threshold: float = 50.0):
        """Automatically detect spatial relationships between objects."""
        for obj1_id, obj1 in self.objects.items():
            for obj2_id, obj2 in self.objects.items():
                if obj1_id != obj2_id:
                    # Calculate centers
                    c1 = obj1.bbox.center
                    c2 = obj2.bbox.center
                    
                    # Calculate distance
                    distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    
                    if distance <= distance_threshold:
                        # Determine spatial relationship
                        dx = c2[0] - c1[0]
                        dy = c2[1] - c1[1]
                        
                        if abs(dx) > abs(dy):
                            relationship = "right_of" if dx > 0 else "left_of"
                        else:
                            relationship = "below" if dy > 0 else "above"
                        
                        self.add_relationship(obj1_id, obj2_id, relationship)
    
    def get_object_relationships(self, object_id: int) -> List[Tuple[int, str]]:
        """Get all relationships for a specific object."""
        relationships = []
        for _, target, data in self.graph.out_edges(object_id, data=True):
            relationships.append((target, data['relationship']))
        return relationships
    
    def visualize(self):
        """Return a visualization of the scene graph using NetworkX."""
        pos = nx.spring_layout(self.graph)
        labels = {node: f"{self.objects[node].class_name}" for node in self.graph.nodes()}
        edge_labels = nx.get_edge_attributes(self.graph, 'relationship')
        return pos, labels, edge_labels

# Example usage
def create_sample_scene():
    scene = SceneGraph()

    model = YOLO("weights/yolov8n.pt")  # initialize model
    results = model("data/image.jpg")  # perform inference
    #print(dir(results))  # print results
    results[0].show()  # display results for the first image

    # get object class and  bounding box and confidence score of each object and make a list of objects
    # Iterate through the detected objects
    objects = []
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes in xyxy format
        scores = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs

        # Convert class IDs to class names
        class_names = [result.names[int(cls_id)] for cls_id in class_ids]

        # Print the details of each detected object
        for i in range(len(boxes)):
            object_id   = i+1
            class_name  = class_names[i]
            bbox        = boxes[i].tolist()
            confidence  = scores[i].item()
            objects.append(SceneObject(object_id, class_name, BoundingBox(*bbox), confidence))
        #print(objects)

    
    # Add objects to scene graph
    for obj in objects:
        scene.add_object(obj)
    
    # Detect spatial relationships
    scene.detect_spatial_relationships()

    # prit scene graph object
    print(scene.objects)
    print(f"nodes :")

    for obj_id, obj in scene.objects.items():
        print(f"{obj_id} : {obj.class_name}")
        print(f"attributes :")
    
    
    return scene



create_scene                = create_sample_scene()
pos, labels, edge_labels    = create_scene.visualize()
# Visualize the graph
plt.figure(figsize=(8, 6))
nx.draw(create_scene.graph, pos, labels=labels, with_labels=True)
nx.draw_networkx_edge_labels(create_scene.graph, pos, edge_labels=edge_labels)
plt.show()