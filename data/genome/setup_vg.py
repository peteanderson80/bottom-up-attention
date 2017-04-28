#!/usr/bin/python


''' Visual genome data analysis and preprocessing.'''

import json
import os
import operator
from visual_genome_python_driver import local as vg
from collections import Counter, defaultdict
import xml.etree.cElementTree as ET
from xml.dom import minidom


dataDir = './data/vg'
outDir = 'data/genome'

# Set maximum values for number of object / attribute / relation classes,
# filter it further later
max_objects = 2500
max_attributes = 1000
max_relations = 500

common_attributes = set(['white','black','blue','green','red','brown','yellow',
    'small','large','silver','wooden','orange','gray','grey','metal','pink','tall',
    'long','dark'])
      
def clean_string(string):
  string = string.lower().strip()
  if len(string) >= 1 and string[-1] == '.':
    return string[:-1].strip()
  return string
  
def clean_objects(string, common_attributes):
  ''' Return object and attribute lists '''
  string = clean_string(string)
  words = string.split()
  if len(words) > 1:
    prefix_words_are_adj = True
    for att in words[:-1]:
      if not att in common_attributes:
        prefix_words_are_adj = False
    if prefix_words_are_adj:
      return words[-1:],words[:-1]
    else:
      return [string],[]
  else:
    return [string],[]
        
def clean_attributes(string):
  ''' Return attribute list '''
  string = clean_string(string)
  if string == "black and white":
    return [string]
  else:
    return [word.lower().strip() for word in string.split(" and ")]

def clean_relations(string):
  string = clean_string(string)
  if len(string) > 0:
    return [string]
  else:
    return []
    
def prettify(elem):
  ''' Return a pretty-printed XML string for the Element '''
  rough_string = ET.tostring(elem, 'utf-8')
  reparsed = minidom.parseString(rough_string)
  return reparsed.toprettyxml(indent="  ")
    
def build_vocabs_and_xml(): 
  objects = Counter()
  attributes = Counter()
  relations = Counter()
  
  with open(os.path.join(dataDir, 'scene_graphs.json')) as f:
    data = json.load(f)
    
  # First extract attributes and relations
  for sg in data:             
    for attr in sg['attributes']:
      try:
        attributes.update(clean_attributes(attr['attribute']['attributes'][0]))
      except:
        pass
    for rel in sg['relationships']:
      relations.update(clean_relations(rel['predicate']))

  # Now extract objects, while looking for common adjectives that will be repurposed
  # as attributes
  for sg in data:    
    for obj in sg['objects']:
      o,a = clean_objects(obj['names'][0], common_attributes)
      objects.update(o)
      attributes.update(a)
  
  with open(os.path.join(outDir, "objects_count.txt"), "w") as text_file:
    for k,v in sorted(objects.iteritems(), key=operator.itemgetter(1), reverse=True):
      text_file.write("%s\t%d\n" % (k.encode('utf-8'),v))
      
  with open(os.path.join(outDir, "attributes_count.txt"), "w") as text_file:
    for k,v in sorted(attributes.iteritems(), key=operator.itemgetter(1), reverse=True):
      text_file.write("%s\t%d\n" % (k.encode('utf-8'),v))
      
  with open(os.path.join(outDir, "relations_count.txt"), "w") as text_file:
    for k,v in sorted(relations.iteritems(), key=operator.itemgetter(1), reverse=True):
      text_file.write("%s\t%d\n" % (k.encode('utf-8'),v))

  # Create full-sized vocabs
  objects = set([k for k,v in objects.most_common(max_objects)])
  attributes = set([k for k,v in attributes.most_common(max_attributes)])
  relations = set([k for k,v in relations.most_common(max_relations)])
  
  with open(os.path.join(outDir, "objects_vocab_%s.txt" % max_objects), "w") as text_file:
    for item in objects:
      text_file.write("%s\n" % item)
  with open(os.path.join(outDir, "attributes_vocab_%s.txt" % max_attributes), "w") as text_file:
    for item in attributes:
      text_file.write("%s\n" % item)
  with open(os.path.join(outDir, "relations_vocab_%s.txt" % max_relations), "w") as text_file:
    for item in relations:
      text_file.write("%s\n" % item)
  

  # Load image metadata
  metadata = {}
  with open(os.path.join(dataDir, 'image_data.json')) as f:
    for item in json.load(f):
      metadata[item['image_id']] = item
  
  # Output clean xml files, one per image
  out_folder = 'xml'
  if not os.path.exists(os.path.join(outDir, out_folder)):
    os.mkdir(os.path.join(outDir, out_folder))
  for sg in data: 
    ann = ET.Element("annotation")
    meta = metadata[sg["image_id"]]
    assert sg["image_id"] == meta["image_id"]
    url_split = meta["url"].split("/")
    ET.SubElement(ann, "folder").text = url_split[-2]
    ET.SubElement(ann, "filename").text = url_split[-1]
  
    source = ET.SubElement(ann, "source")
    ET.SubElement(source, "database").text = "Visual Genome Version 1.2"
    ET.SubElement(source, "image_id").text = str(meta["image_id"])
    ET.SubElement(source, "coco_id").text = str(meta["coco_id"])
    ET.SubElement(source, "flickr_id").text = str(meta["flickr_id"])
    
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(meta["width"])
    ET.SubElement(size, "height").text = str(meta["height"])
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(ann, "segmented").text = "0"
    
    object_set = set()
    for obj in sg['objects']:
      o,a = clean_objects(obj['names'][0], common_attributes)
      if o[0] in objects:
        ob = ET.SubElement(ann, "object")
        ET.SubElement(ob, "name").text = o[0]
        ET.SubElement(ob, "object_id").text = str(obj["object_id"])
        object_set.add(obj["object_id"])
        ET.SubElement(ob, "difficult").text = "0"
        bbox = ET.SubElement(ob, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(obj["x"])
        ET.SubElement(bbox, "ymin").text = str(obj["y"])
        ET.SubElement(bbox, "xmax").text = str(obj["x"] + obj["w"])
        ET.SubElement(bbox, "ymax").text = str(obj["y"] + obj["h"])
        attribute_set = set()
        for attribute_name in a:
          if attribute_name in attributes:
            attribute_set.add(attribute_name)
        for attr in sg['attributes']:
          if attr["attribute"]["object_id"] == obj["object_id"]:
            try:
              for ix in attr['attribute']['attributes']:
                for clean_attribute in clean_attributes(ix):
                  if clean_attribute in attributes:
                    attribute_set.add(clean_attribute)
            except:
              pass
        for attribute_name in attribute_set:
          ET.SubElement(ob, "attribute").text = attribute_name
          
    for rel in sg['relationships']:
      predicate = clean_string(rel["predicate"])
      if rel["subject_id"] in object_set and rel["object_id"] in object_set:
        if predicate in relations:
          re = ET.SubElement(ann, "relation")
          ET.SubElement(re, "subject_id").text = str(rel["subject_id"])
          ET.SubElement(re, "object_id").text = str(rel["object_id"])
          ET.SubElement(re, "predicate").text = predicate
    
    outFile = url_split[-1].replace(".jpg",".xml")
    tree = ET.ElementTree(ann)
    if len(tree.findall('object')) > 0:
      tree.write(os.path.join(outDir, out_folder, outFile))
    

  
if __name__ == "__main__":

  # First, use visual genome library to merge attributes and scene graphs
  vg.AddAttrsToSceneGraphs(dataDir=dataDir)
  # Next, build xml files
  build_vocabs_and_xml()
  
  
