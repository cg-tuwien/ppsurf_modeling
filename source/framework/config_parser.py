from os import path
import sys
from urllib.error import URLError
from xml.etree import ElementTree
from xmlschema import XMLSchema11, XMLSchemaValidationError
import networkx

from source.framework import global_config
from source.framework.utility import Edge, IOType


class ConfigParser:
    def __init__(self, config, schema, user_feedback=sys.stdout, error_log=sys.stderr):
        self._config_schema = schema
        self._config_location = config
        self._user_feedback = user_feedback
        self._error_log = error_log
        self._validated = False

    def set_config(self, config):
        self._config_location = config
        self._validated = False

    def set_schema(self, schema):
        self._config_schema = schema
        self._validated = False

    def validate_config(self):
        if self._validated:
            return True

        success = True

        try:
            # Create new XML Schema v.1.1
            xml_schema = XMLSchema11(self._config_schema)
            # Validate config using given schema
            xml_schema.validate(self._config_location)
        except URLError as err:
            print("File not found: " + str(err.reason.filename) + "\n", file=self._user_feedback)
            success = False
        except XMLSchemaValidationError as err:
            print("XML Schema validation error!\n", file=self._user_feedback)
            print(err, file=self._error_log)
            success = False
        except ElementTree.ParseError as err:
            print("Parse error while reading config!\n", file=self._user_feedback)
            print(err, file=self._error_log)
            success = False
        except Exception as err:
            print("Unknown error!\n", file=self._user_feedback)
            print(err, file=self._error_log)
            success = False

        self._validated = success
        return success

    def parse_config(self):
        def insert_directory(node):
            complete_path = node.text

            if (node.tag == "option" or node.tag == "argument") and "value" in node.attrib:
                complete_path = node.attrib["value"]

            if "useFolder" in node.attrib:
                folder = user_defined_folders[node.attrib["useFolder"]]
                if node.tag == "inputDir" or node.tag == "outputDir":
                    complete_path = path.join(folder, complete_path)
                elif (node.tag == "call" or node.tag == "argument" or node.tag == "option") and "{}" in complete_path:
                    complete_path = complete_path.format(folder)

            return complete_path

        # Create new directed graph
        graph = networkx.DiGraph()

        # Parse XML configuration into element tree
        config_tree = ElementTree.parse(self._config_location)

        # Get root element of the configuration tree
        root_node = config_tree.getroot()

        # Extract configuration version
        version = config_tree.getroot().attrib["version"]
        try:
            version = float(version)
        except ValueError:
            pass
        else:
            # Set version number in global configuration
            global_config.set_version(version)

        # Check if version is compatible
        if version not in (0.1, 0.2):
            print("Unknown schema version!\n", file=self._user_feedback)
            return None

        user_defined_folders = {}
        config_node = root_node.find("configuration")
        if config_node:
            for folder_node in config_node:
                user_defined_folders[folder_node.attrib["name"]] = folder_node.text

        # Iterate over every edge definition in configuration
        for edge_xml in root_node:
            if edge_xml.tag == "configuration":
                continue

            # Start concatenating call string
            call = insert_directory(edge_xml.find("call"))
            format_order = []

            inputs = [insert_directory(e) for e in edge_xml.findall("inputDir")]
            outputs = [insert_directory(e) for e in edge_xml.findall("outputDir")]

            # Traverse every specified parameter of the call
            for p in edge_xml.find("parameters"):
                if p.tag == "option":
                    # Options always start with the name of the flag (or is only the name)
                    call += " " + p.attrib["name"]

                # If a value is given then append it to the call
                if "value" in p.attrib and "type" not in p.attrib:
                    call += " " + insert_directory(p)
                # If a type instead of a value is given append a placeholder...
                elif "type" in p.attrib and "value" not in p.attrib:
                    call += " {}"
                    # ... and add the type of placeholder to the format order
                    format_order.append((IOType.INPUT, inputs.pop(0)) if p.attrib["type"] == "input" else (IOType.OUTPUT, outputs.pop(0)))

            # Create edge object which holds information about it
            edge_info = Edge(edge_xml.find("name").text, edge_xml.find("description").text, call, format_order, edge_xml.attrib["type"])

            if version >= 0.2 and "require_complete" in edge_xml.attrib:
                require_complete = bool(edge_xml.attrib["require_complete"])
                edge_info.set_require_complete(require_complete)

            # Create edges in graph with input- and output-directories as nodes
            for input_directory in edge_xml.findall("inputDir"):
                for output_directory in edge_xml.findall("outputDir"):
                    graph.add_edge(input_directory.text, output_directory.text, info=edge_info, label=edge_info.get_name())

        # Check for cycles in the graph
        try:
            cycles = networkx.find_cycle(graph)
        except networkx.NetworkXNoCycle:
            pass
        else:
            # Found cycles in graph
            cycle_edges = []
            for c in cycles:
                cycle_edges.append(graph.edges[c]["info"].get_name())
            print("Found cycle in pipeline graph at following edges: {}".format(cycle_edges))
            return None

        return graph
