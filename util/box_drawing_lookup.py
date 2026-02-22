#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Box Drawing and Block Element Character Lookup Utility
#
# Author:        Paul Calnon
# Version:       1.0.0
# File Name:     box_drawing_lookup.py
# File Path:     juniper-cascor/util/
#
# Date Created:  2026-02-22
# Last Modified: 2026-02-22
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     Comprehensive lookup utility for Unicode Box Drawing characters (U+2500-U+257F),
#     Block Elements (U+2580-U+259F), and their Code Page 437 (CP437/IBM PC) equivalents.
#
#     Features:
#       - Look up characters by Unicode code point, CP437 code, decimal value, or description
#       - Generate markdown tables for documentation
#       - Display visual box drawing examples
#       - Cross-reference between Unicode and CP437 encodings
#
#     Usage:
#       python box_drawing_lookup.py                    # Print all tables
#       python box_drawing_lookup.py --search "corner"  # Search by description
#       python box_drawing_lookup.py --cp437             # Show CP437 mapping table
#       python box_drawing_lookup.py --unicode           # Show full Unicode box drawing table
#       python box_drawing_lookup.py --blocks            # Show block elements table
#       python box_drawing_lookup.py --examples          # Show visual box drawing examples
#       python box_drawing_lookup.py --lookup U+2554     # Look up specific code point
#       python box_drawing_lookup.py --lookup 201        # Look up by CP437 decimal
#       python box_drawing_lookup.py --markdown          # Output tables in markdown format
#       python box_drawing_lookup.py --markdown-full     # Output the complete markdown document
#
#####################################################################################################################################################################################################
# Notes:
#     CP437 (Code Page 437) is the original IBM PC character set, also known as DOS Latin US.
#     Characters 128-255 in CP437 include box drawing characters that were widely used in
#     text-mode DOS applications. These map to various Unicode code points, primarily in the
#     Box Drawing (U+2500-U+257F) and Block Elements (U+2580-U+259F) blocks.
#
#####################################################################################################################################################################################################
# References:
#     - Unicode Standard, Chapter 22: Symbols — Box Drawing
#       https://www.unicode.org/charts/PDF/U2500.pdf
#     - Unicode Standard — Block Elements
#       https://www.unicode.org/charts/PDF/U2580.pdf
#     - Code Page 437 (IBM PC)
#       https://en.wikipedia.org/wiki/Code_page_437
#     - Fahlman, S.E. & Lebiere, C. (1990). The Cascade-Correlation Learning Architecture.
#
#####################################################################################################################################################################################################

import argparse
import sys
from dataclasses import dataclass


# ===================================================================
# Data Classes
# ===================================================================

@dataclass
class BoxChar:
    """Represents a box drawing or block element character with all encoding info."""
    char: str
    unicode_hex: str        # e.g., "2554"
    unicode_dec: int        # e.g., 9556
    unicode_name: str       # e.g., "BOX DRAWINGS DOUBLE DOWN AND RIGHT"
    cp437_dec: int = None   # e.g., 201 (None if no CP437 equivalent)
    category: str = ""      # Grouping category

    @property
    def cp437_hex(self) -> str:
        return f"0x{self.cp437_dec:02X}" if self.cp437_dec is not None else ""

    @property
    def html_hex(self) -> str:
        return f"&#x{self.unicode_hex};"

    @property
    def html_dec(self) -> str:
        return f"&#{self.unicode_dec};"

    @property
    def unicode_codepoint(self) -> str:
        return f"U+{self.unicode_hex}"


# ===================================================================
# Complete CP437 Box Drawing / Block Element Mapping (128-255 relevant subset)
# ===================================================================

CP437_BOX_DRAWING = [
    # CP437 Dec -> (Unicode char, Unicode hex, Unicode dec, Unicode name)
    BoxChar("░", "2591", 9617, "LIGHT SHADE", 176, "Shade Blocks"),
    BoxChar("▒", "2592", 9618, "MEDIUM SHADE", 177, "Shade Blocks"),
    BoxChar("▓", "2593", 9619, "DARK SHADE", 178, "Shade Blocks"),
    BoxChar("│", "2502", 9474, "BOX DRAWINGS LIGHT VERTICAL", 179, "Light Lines"),
    BoxChar("┤", "2524", 9508, "BOX DRAWINGS LIGHT VERTICAL AND LEFT", 180, "Light T-Junctions"),
    BoxChar("╡", "2561", 9569, "BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE", 181, "Single/Double Transitions"),
    BoxChar("╢", "2562", 9570, "BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE", 182, "Single/Double Transitions"),
    BoxChar("╖", "2556", 9558, "BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE", 183, "Single/Double Transitions"),
    BoxChar("╕", "2555", 9557, "BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE", 184, "Single/Double Transitions"),
    BoxChar("╣", "2563", 9571, "BOX DRAWINGS DOUBLE VERTICAL AND LEFT", 185, "Double T-Junctions"),
    BoxChar("║", "2551", 9553, "BOX DRAWINGS DOUBLE VERTICAL", 186, "Double Lines"),
    BoxChar("╗", "2557", 9559, "BOX DRAWINGS DOUBLE DOWN AND LEFT", 187, "Double Corners"),
    BoxChar("╝", "255D", 9565, "BOX DRAWINGS DOUBLE UP AND LEFT", 188, "Double Corners"),
    BoxChar("╜", "255C", 9564, "BOX DRAWINGS UP DOUBLE AND LEFT SINGLE", 189, "Single/Double Transitions"),
    BoxChar("╛", "255B", 9563, "BOX DRAWINGS UP SINGLE AND LEFT DOUBLE", 190, "Single/Double Transitions"),
    BoxChar("┐", "2510", 9488, "BOX DRAWINGS LIGHT DOWN AND LEFT", 191, "Light Corners"),
    BoxChar("└", "2514", 9492, "BOX DRAWINGS LIGHT UP AND RIGHT", 192, "Light Corners"),
    BoxChar("┴", "2534", 9524, "BOX DRAWINGS LIGHT UP AND HORIZONTAL", 193, "Light T-Junctions"),
    BoxChar("┬", "252C", 9516, "BOX DRAWINGS LIGHT DOWN AND HORIZONTAL", 194, "Light T-Junctions"),
    BoxChar("├", "251C", 9500, "BOX DRAWINGS LIGHT VERTICAL AND RIGHT", 195, "Light T-Junctions"),
    BoxChar("─", "2500", 9472, "BOX DRAWINGS LIGHT HORIZONTAL", 196, "Light Lines"),
    BoxChar("┼", "253C", 9532, "BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL", 197, "Light Crosses"),
    BoxChar("╞", "255E", 9566, "BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE", 198, "Single/Double Transitions"),
    BoxChar("╟", "255F", 9567, "BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE", 199, "Single/Double Transitions"),
    BoxChar("╚", "255A", 9562, "BOX DRAWINGS DOUBLE UP AND RIGHT", 200, "Double Corners"),
    BoxChar("╔", "2554", 9556, "BOX DRAWINGS DOUBLE DOWN AND RIGHT", 201, "Double Corners"),
    BoxChar("╩", "2569", 9577, "BOX DRAWINGS DOUBLE UP AND HORIZONTAL", 202, "Double T-Junctions"),
    BoxChar("╦", "2566", 9574, "BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL", 203, "Double T-Junctions"),
    BoxChar("╠", "2560", 9568, "BOX DRAWINGS DOUBLE VERTICAL AND RIGHT", 204, "Double T-Junctions"),
    BoxChar("═", "2550", 9552, "BOX DRAWINGS DOUBLE HORIZONTAL", 205, "Double Lines"),
    BoxChar("╬", "256C", 9580, "BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL", 206, "Double Crosses"),
    BoxChar("╧", "2567", 9575, "BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE", 207, "Single/Double Transitions"),
    BoxChar("╨", "2568", 9576, "BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE", 208, "Single/Double Transitions"),
    BoxChar("╤", "2564", 9572, "BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE", 209, "Single/Double Transitions"),
    BoxChar("╥", "2565", 9573, "BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE", 210, "Single/Double Transitions"),
    BoxChar("╙", "2559", 9561, "BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE", 211, "Single/Double Transitions"),
    BoxChar("╘", "2558", 9560, "BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE", 212, "Single/Double Transitions"),
    BoxChar("╒", "2552", 9554, "BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE", 213, "Single/Double Transitions"),
    BoxChar("╓", "2553", 9555, "BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE", 214, "Single/Double Transitions"),
    BoxChar("╫", "256B", 9579, "BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE", 215, "Single/Double Transitions"),
    BoxChar("╪", "256A", 9578, "BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE", 216, "Single/Double Transitions"),
    BoxChar("┘", "2518", 9496, "BOX DRAWINGS LIGHT UP AND LEFT", 217, "Light Corners"),
    BoxChar("┌", "250C", 9484, "BOX DRAWINGS LIGHT DOWN AND RIGHT", 218, "Light Corners"),
    BoxChar("█", "2588", 9608, "FULL BLOCK", 219, "Block Elements"),
    BoxChar("▄", "2584", 9604, "LOWER HALF BLOCK", 220, "Block Elements"),
    BoxChar("▌", "258C", 9612, "LEFT HALF BLOCK", 221, "Block Elements"),
    BoxChar("▐", "2590", 9616, "RIGHT HALF BLOCK", 222, "Block Elements"),
    BoxChar("▀", "2580", 9600, "UPPER HALF BLOCK", 223, "Block Elements"),
]


# ===================================================================
# Complete Unicode Box Drawing Block (U+2500 - U+257F) — 128 characters
# ===================================================================

UNICODE_BOX_DRAWING = [
    # Lines - Light and Heavy
    BoxChar("─", "2500", 9472, "BOX DRAWINGS LIGHT HORIZONTAL", 196, "Lines"),
    BoxChar("━", "2501", 9473, "BOX DRAWINGS HEAVY HORIZONTAL", None, "Lines"),
    BoxChar("│", "2502", 9474, "BOX DRAWINGS LIGHT VERTICAL", 179, "Lines"),
    BoxChar("┃", "2503", 9475, "BOX DRAWINGS HEAVY VERTICAL", None, "Lines"),
    # Dashed - Triple Dash
    BoxChar("┄", "2504", 9476, "BOX DRAWINGS LIGHT TRIPLE DASH HORIZONTAL", None, "Dashed Lines"),
    BoxChar("┅", "2505", 9477, "BOX DRAWINGS HEAVY TRIPLE DASH HORIZONTAL", None, "Dashed Lines"),
    BoxChar("┆", "2506", 9478, "BOX DRAWINGS LIGHT TRIPLE DASH VERTICAL", None, "Dashed Lines"),
    BoxChar("┇", "2507", 9479, "BOX DRAWINGS HEAVY TRIPLE DASH VERTICAL", None, "Dashed Lines"),
    # Dashed - Quadruple Dash
    BoxChar("┈", "2508", 9480, "BOX DRAWINGS LIGHT QUADRUPLE DASH HORIZONTAL", None, "Dashed Lines"),
    BoxChar("┉", "2509", 9481, "BOX DRAWINGS HEAVY QUADRUPLE DASH HORIZONTAL", None, "Dashed Lines"),
    BoxChar("┊", "250A", 9482, "BOX DRAWINGS LIGHT QUADRUPLE DASH VERTICAL", None, "Dashed Lines"),
    BoxChar("┋", "250B", 9483, "BOX DRAWINGS HEAVY QUADRUPLE DASH VERTICAL", None, "Dashed Lines"),
    # Corners - Down and Right
    BoxChar("┌", "250C", 9484, "BOX DRAWINGS LIGHT DOWN AND RIGHT", 218, "Corners"),
    BoxChar("┍", "250D", 9485, "BOX DRAWINGS DOWN LIGHT AND RIGHT HEAVY", None, "Corners"),
    BoxChar("┎", "250E", 9486, "BOX DRAWINGS DOWN HEAVY AND RIGHT LIGHT", None, "Corners"),
    BoxChar("┏", "250F", 9487, "BOX DRAWINGS HEAVY DOWN AND RIGHT", None, "Corners"),
    # Corners - Down and Left
    BoxChar("┐", "2510", 9488, "BOX DRAWINGS LIGHT DOWN AND LEFT", 191, "Corners"),
    BoxChar("┑", "2511", 9489, "BOX DRAWINGS DOWN LIGHT AND LEFT HEAVY", None, "Corners"),
    BoxChar("┒", "2512", 9490, "BOX DRAWINGS DOWN HEAVY AND LEFT LIGHT", None, "Corners"),
    BoxChar("┓", "2513", 9491, "BOX DRAWINGS HEAVY DOWN AND LEFT", None, "Corners"),
    # Corners - Up and Right
    BoxChar("└", "2514", 9492, "BOX DRAWINGS LIGHT UP AND RIGHT", 192, "Corners"),
    BoxChar("┕", "2515", 9493, "BOX DRAWINGS UP LIGHT AND RIGHT HEAVY", None, "Corners"),
    BoxChar("┖", "2516", 9494, "BOX DRAWINGS UP HEAVY AND RIGHT LIGHT", None, "Corners"),
    BoxChar("┗", "2517", 9495, "BOX DRAWINGS HEAVY UP AND RIGHT", None, "Corners"),
    # Corners - Up and Left
    BoxChar("┘", "2518", 9496, "BOX DRAWINGS LIGHT UP AND LEFT", 217, "Corners"),
    BoxChar("┙", "2519", 9497, "BOX DRAWINGS UP LIGHT AND LEFT HEAVY", None, "Corners"),
    BoxChar("┚", "251A", 9498, "BOX DRAWINGS UP HEAVY AND LEFT LIGHT", None, "Corners"),
    BoxChar("┛", "251B", 9499, "BOX DRAWINGS HEAVY UP AND LEFT", None, "Corners"),
    # T-Junctions - Vertical and Right
    BoxChar("├", "251C", 9500, "BOX DRAWINGS LIGHT VERTICAL AND RIGHT", 195, "T-Junctions"),
    BoxChar("┝", "251D", 9501, "BOX DRAWINGS VERTICAL LIGHT AND RIGHT HEAVY", None, "T-Junctions"),
    BoxChar("┞", "251E", 9502, "BOX DRAWINGS UP HEAVY AND RIGHT DOWN LIGHT", None, "T-Junctions"),
    BoxChar("┟", "251F", 9503, "BOX DRAWINGS DOWN HEAVY AND RIGHT UP LIGHT", None, "T-Junctions"),
    BoxChar("┠", "2520", 9504, "BOX DRAWINGS VERTICAL HEAVY AND RIGHT LIGHT", None, "T-Junctions"),
    BoxChar("┡", "2521", 9505, "BOX DRAWINGS DOWN LIGHT AND RIGHT UP HEAVY", None, "T-Junctions"),
    BoxChar("┢", "2522", 9506, "BOX DRAWINGS UP LIGHT AND RIGHT DOWN HEAVY", None, "T-Junctions"),
    BoxChar("┣", "2523", 9507, "BOX DRAWINGS HEAVY VERTICAL AND RIGHT", None, "T-Junctions"),
    # T-Junctions - Vertical and Left
    BoxChar("┤", "2524", 9508, "BOX DRAWINGS LIGHT VERTICAL AND LEFT", 180, "T-Junctions"),
    BoxChar("┥", "2525", 9509, "BOX DRAWINGS VERTICAL LIGHT AND LEFT HEAVY", None, "T-Junctions"),
    BoxChar("┦", "2526", 9510, "BOX DRAWINGS UP HEAVY AND LEFT DOWN LIGHT", None, "T-Junctions"),
    BoxChar("┧", "2527", 9511, "BOX DRAWINGS DOWN HEAVY AND LEFT UP LIGHT", None, "T-Junctions"),
    BoxChar("┨", "2528", 9512, "BOX DRAWINGS VERTICAL HEAVY AND LEFT LIGHT", None, "T-Junctions"),
    BoxChar("┩", "2529", 9513, "BOX DRAWINGS DOWN LIGHT AND LEFT UP HEAVY", None, "T-Junctions"),
    BoxChar("┪", "252A", 9514, "BOX DRAWINGS UP LIGHT AND LEFT DOWN HEAVY", None, "T-Junctions"),
    BoxChar("┫", "252B", 9515, "BOX DRAWINGS HEAVY VERTICAL AND LEFT", None, "T-Junctions"),
    # T-Junctions - Down and Horizontal
    BoxChar("┬", "252C", 9516, "BOX DRAWINGS LIGHT DOWN AND HORIZONTAL", 194, "T-Junctions"),
    BoxChar("┭", "252D", 9517, "BOX DRAWINGS LEFT HEAVY AND RIGHT DOWN LIGHT", None, "T-Junctions"),
    BoxChar("┮", "252E", 9518, "BOX DRAWINGS RIGHT HEAVY AND LEFT DOWN LIGHT", None, "T-Junctions"),
    BoxChar("┯", "252F", 9519, "BOX DRAWINGS DOWN LIGHT AND HORIZONTAL HEAVY", None, "T-Junctions"),
    BoxChar("┰", "2530", 9520, "BOX DRAWINGS DOWN HEAVY AND HORIZONTAL LIGHT", None, "T-Junctions"),
    BoxChar("┱", "2531", 9521, "BOX DRAWINGS RIGHT LIGHT AND LEFT DOWN HEAVY", None, "T-Junctions"),
    BoxChar("┲", "2532", 9522, "BOX DRAWINGS LEFT LIGHT AND RIGHT DOWN HEAVY", None, "T-Junctions"),
    BoxChar("┳", "2533", 9523, "BOX DRAWINGS HEAVY DOWN AND HORIZONTAL", None, "T-Junctions"),
    # T-Junctions - Up and Horizontal
    BoxChar("┴", "2534", 9524, "BOX DRAWINGS LIGHT UP AND HORIZONTAL", 193, "T-Junctions"),
    BoxChar("┵", "2535", 9525, "BOX DRAWINGS LEFT HEAVY AND RIGHT UP LIGHT", None, "T-Junctions"),
    BoxChar("┶", "2536", 9526, "BOX DRAWINGS RIGHT HEAVY AND LEFT UP LIGHT", None, "T-Junctions"),
    BoxChar("┷", "2537", 9527, "BOX DRAWINGS UP LIGHT AND HORIZONTAL HEAVY", None, "T-Junctions"),
    BoxChar("┸", "2538", 9528, "BOX DRAWINGS UP HEAVY AND HORIZONTAL LIGHT", None, "T-Junctions"),
    BoxChar("┹", "2539", 9529, "BOX DRAWINGS RIGHT LIGHT AND LEFT UP HEAVY", None, "T-Junctions"),
    BoxChar("┺", "253A", 9530, "BOX DRAWINGS LEFT LIGHT AND RIGHT UP HEAVY", None, "T-Junctions"),
    BoxChar("┻", "253B", 9531, "BOX DRAWINGS HEAVY UP AND HORIZONTAL", None, "T-Junctions"),
    # Crosses
    BoxChar("┼", "253C", 9532, "BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL", 197, "Crosses"),
    BoxChar("┽", "253D", 9533, "BOX DRAWINGS LEFT HEAVY AND RIGHT VERTICAL LIGHT", None, "Crosses"),
    BoxChar("┾", "253E", 9534, "BOX DRAWINGS RIGHT HEAVY AND LEFT VERTICAL LIGHT", None, "Crosses"),
    BoxChar("┿", "253F", 9535, "BOX DRAWINGS VERTICAL LIGHT AND HORIZONTAL HEAVY", None, "Crosses"),
    BoxChar("╀", "2540", 9536, "BOX DRAWINGS UP HEAVY AND DOWN HORIZONTAL LIGHT", None, "Crosses"),
    BoxChar("╁", "2541", 9537, "BOX DRAWINGS DOWN HEAVY AND UP HORIZONTAL LIGHT", None, "Crosses"),
    BoxChar("╂", "2542", 9538, "BOX DRAWINGS VERTICAL HEAVY AND HORIZONTAL LIGHT", None, "Crosses"),
    BoxChar("╃", "2543", 9539, "BOX DRAWINGS LEFT UP HEAVY AND RIGHT DOWN LIGHT", None, "Crosses"),
    BoxChar("╄", "2544", 9540, "BOX DRAWINGS RIGHT UP HEAVY AND LEFT DOWN LIGHT", None, "Crosses"),
    BoxChar("╅", "2545", 9541, "BOX DRAWINGS LEFT DOWN HEAVY AND RIGHT UP LIGHT", None, "Crosses"),
    BoxChar("╆", "2546", 9542, "BOX DRAWINGS RIGHT DOWN HEAVY AND LEFT UP LIGHT", None, "Crosses"),
    BoxChar("╇", "2547", 9543, "BOX DRAWINGS DOWN LIGHT AND UP HORIZONTAL HEAVY", None, "Crosses"),
    BoxChar("╈", "2548", 9544, "BOX DRAWINGS UP LIGHT AND DOWN HORIZONTAL HEAVY", None, "Crosses"),
    BoxChar("╉", "2549", 9545, "BOX DRAWINGS RIGHT LIGHT AND LEFT VERTICAL HEAVY", None, "Crosses"),
    BoxChar("╊", "254A", 9546, "BOX DRAWINGS LEFT LIGHT AND RIGHT VERTICAL HEAVY", None, "Crosses"),
    BoxChar("╋", "254B", 9547, "BOX DRAWINGS HEAVY VERTICAL AND HORIZONTAL", None, "Crosses"),
    # Double Dash
    BoxChar("╌", "254C", 9548, "BOX DRAWINGS LIGHT DOUBLE DASH HORIZONTAL", None, "Dashed Lines"),
    BoxChar("╍", "254D", 9549, "BOX DRAWINGS HEAVY DOUBLE DASH HORIZONTAL", None, "Dashed Lines"),
    BoxChar("╎", "254E", 9550, "BOX DRAWINGS LIGHT DOUBLE DASH VERTICAL", None, "Dashed Lines"),
    BoxChar("╏", "254F", 9551, "BOX DRAWINGS HEAVY DOUBLE DASH VERTICAL", None, "Dashed Lines"),
    # Double Lines
    BoxChar("═", "2550", 9552, "BOX DRAWINGS DOUBLE HORIZONTAL", 205, "Double Lines"),
    BoxChar("║", "2551", 9553, "BOX DRAWINGS DOUBLE VERTICAL", 186, "Double Lines"),
    # Double/Single Transition Corners - Down and Right
    BoxChar("╒", "2552", 9554, "BOX DRAWINGS DOWN SINGLE AND RIGHT DOUBLE", 213, "Single/Double Corners"),
    BoxChar("╓", "2553", 9555, "BOX DRAWINGS DOWN DOUBLE AND RIGHT SINGLE", 214, "Single/Double Corners"),
    BoxChar("╔", "2554", 9556, "BOX DRAWINGS DOUBLE DOWN AND RIGHT", 201, "Double Corners"),
    # Double/Single Transition Corners - Down and Left
    BoxChar("╕", "2555", 9557, "BOX DRAWINGS DOWN SINGLE AND LEFT DOUBLE", 184, "Single/Double Corners"),
    BoxChar("╖", "2556", 9558, "BOX DRAWINGS DOWN DOUBLE AND LEFT SINGLE", 183, "Single/Double Corners"),
    BoxChar("╗", "2557", 9559, "BOX DRAWINGS DOUBLE DOWN AND LEFT", 187, "Double Corners"),
    # Double/Single Transition Corners - Up and Right
    BoxChar("╘", "2558", 9560, "BOX DRAWINGS UP SINGLE AND RIGHT DOUBLE", 212, "Single/Double Corners"),
    BoxChar("╙", "2559", 9561, "BOX DRAWINGS UP DOUBLE AND RIGHT SINGLE", 211, "Single/Double Corners"),
    BoxChar("╚", "255A", 9562, "BOX DRAWINGS DOUBLE UP AND RIGHT", 200, "Double Corners"),
    # Double/Single Transition Corners - Up and Left
    BoxChar("╛", "255B", 9563, "BOX DRAWINGS UP SINGLE AND LEFT DOUBLE", 190, "Single/Double Corners"),
    BoxChar("╜", "255C", 9564, "BOX DRAWINGS UP DOUBLE AND LEFT SINGLE", 189, "Single/Double Corners"),
    BoxChar("╝", "255D", 9565, "BOX DRAWINGS DOUBLE UP AND LEFT", 188, "Double Corners"),
    # Double/Single Transition T-Junctions
    BoxChar("╞", "255E", 9566, "BOX DRAWINGS VERTICAL SINGLE AND RIGHT DOUBLE", 198, "Single/Double T-Junctions"),
    BoxChar("╟", "255F", 9567, "BOX DRAWINGS VERTICAL DOUBLE AND RIGHT SINGLE", 199, "Single/Double T-Junctions"),
    BoxChar("╠", "2560", 9568, "BOX DRAWINGS DOUBLE VERTICAL AND RIGHT", 204, "Double T-Junctions"),
    BoxChar("╡", "2561", 9569, "BOX DRAWINGS VERTICAL SINGLE AND LEFT DOUBLE", 181, "Single/Double T-Junctions"),
    BoxChar("╢", "2562", 9570, "BOX DRAWINGS VERTICAL DOUBLE AND LEFT SINGLE", 182, "Single/Double T-Junctions"),
    BoxChar("╣", "2563", 9571, "BOX DRAWINGS DOUBLE VERTICAL AND LEFT", 185, "Double T-Junctions"),
    BoxChar("╤", "2564", 9572, "BOX DRAWINGS DOWN SINGLE AND HORIZONTAL DOUBLE", 209, "Single/Double T-Junctions"),
    BoxChar("╥", "2565", 9573, "BOX DRAWINGS DOWN DOUBLE AND HORIZONTAL SINGLE", 210, "Single/Double T-Junctions"),
    BoxChar("╦", "2566", 9574, "BOX DRAWINGS DOUBLE DOWN AND HORIZONTAL", 203, "Double T-Junctions"),
    BoxChar("╧", "2567", 9575, "BOX DRAWINGS UP SINGLE AND HORIZONTAL DOUBLE", 207, "Single/Double T-Junctions"),
    BoxChar("╨", "2568", 9576, "BOX DRAWINGS UP DOUBLE AND HORIZONTAL SINGLE", 208, "Single/Double T-Junctions"),
    BoxChar("╩", "2569", 9577, "BOX DRAWINGS DOUBLE UP AND HORIZONTAL", 202, "Double T-Junctions"),
    # Double/Single Transition Crosses
    BoxChar("╪", "256A", 9578, "BOX DRAWINGS VERTICAL SINGLE AND HORIZONTAL DOUBLE", 216, "Single/Double Crosses"),
    BoxChar("╫", "256B", 9579, "BOX DRAWINGS VERTICAL DOUBLE AND HORIZONTAL SINGLE", 215, "Single/Double Crosses"),
    BoxChar("╬", "256C", 9580, "BOX DRAWINGS DOUBLE VERTICAL AND HORIZONTAL", 206, "Double Crosses"),
    # Arc Corners
    BoxChar("╭", "256D", 9581, "BOX DRAWINGS LIGHT ARC DOWN AND RIGHT", None, "Arc Corners"),
    BoxChar("╮", "256E", 9582, "BOX DRAWINGS LIGHT ARC DOWN AND LEFT", None, "Arc Corners"),
    BoxChar("╯", "256F", 9583, "BOX DRAWINGS LIGHT ARC UP AND LEFT", None, "Arc Corners"),
    BoxChar("╰", "2570", 9584, "BOX DRAWINGS LIGHT ARC UP AND RIGHT", None, "Arc Corners"),
    # Diagonal Lines
    BoxChar("╱", "2571", 9585, "BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT", None, "Diagonals"),
    BoxChar("╲", "2572", 9586, "BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT", None, "Diagonals"),
    BoxChar("╳", "2573", 9587, "BOX DRAWINGS LIGHT DIAGONAL CROSS", None, "Diagonals"),
    # Half-Line Segments - Light
    BoxChar("╴", "2574", 9588, "BOX DRAWINGS LIGHT LEFT", None, "Half Lines"),
    BoxChar("╵", "2575", 9589, "BOX DRAWINGS LIGHT UP", None, "Half Lines"),
    BoxChar("╶", "2576", 9590, "BOX DRAWINGS LIGHT RIGHT", None, "Half Lines"),
    BoxChar("╷", "2577", 9591, "BOX DRAWINGS LIGHT DOWN", None, "Half Lines"),
    # Half-Line Segments - Heavy
    BoxChar("╸", "2578", 9592, "BOX DRAWINGS HEAVY LEFT", None, "Half Lines"),
    BoxChar("╹", "2579", 9593, "BOX DRAWINGS HEAVY UP", None, "Half Lines"),
    BoxChar("╺", "257A", 9594, "BOX DRAWINGS HEAVY RIGHT", None, "Half Lines"),
    BoxChar("╻", "257B", 9595, "BOX DRAWINGS HEAVY DOWN", None, "Half Lines"),
    # Light-Heavy Transitions
    BoxChar("╼", "257C", 9596, "BOX DRAWINGS LIGHT LEFT AND HEAVY RIGHT", None, "Weight Transitions"),
    BoxChar("╽", "257D", 9597, "BOX DRAWINGS LIGHT UP AND HEAVY DOWN", None, "Weight Transitions"),
    BoxChar("╾", "257E", 9598, "BOX DRAWINGS HEAVY LEFT AND LIGHT RIGHT", None, "Weight Transitions"),
    BoxChar("╿", "257F", 9599, "BOX DRAWINGS HEAVY UP AND LIGHT DOWN", None, "Weight Transitions"),
]


# ===================================================================
# Block Elements (U+2580 - U+259F) — 32 characters
# ===================================================================

BLOCK_ELEMENTS = [
    BoxChar("▀", "2580", 9600, "UPPER HALF BLOCK", 223, "Half Blocks"),
    BoxChar("▁", "2581", 9601, "LOWER ONE EIGHTH BLOCK", None, "Fractional Blocks"),
    BoxChar("▂", "2582", 9602, "LOWER ONE QUARTER BLOCK", None, "Fractional Blocks"),
    BoxChar("▃", "2583", 9603, "LOWER THREE EIGHTHS BLOCK", None, "Fractional Blocks"),
    BoxChar("▄", "2584", 9604, "LOWER HALF BLOCK", 220, "Half Blocks"),
    BoxChar("▅", "2585", 9605, "LOWER FIVE EIGHTHS BLOCK", None, "Fractional Blocks"),
    BoxChar("▆", "2586", 9606, "LOWER THREE QUARTERS BLOCK", None, "Fractional Blocks"),
    BoxChar("▇", "2587", 9607, "LOWER SEVEN EIGHTHS BLOCK", None, "Fractional Blocks"),
    BoxChar("█", "2588", 9608, "FULL BLOCK", 219, "Full Block"),
    BoxChar("▉", "2589", 9609, "LEFT SEVEN EIGHTHS BLOCK", None, "Fractional Blocks"),
    BoxChar("▊", "258A", 9610, "LEFT THREE QUARTERS BLOCK", None, "Fractional Blocks"),
    BoxChar("▋", "258B", 9611, "LEFT FIVE EIGHTHS BLOCK", None, "Fractional Blocks"),
    BoxChar("▌", "258C", 9612, "LEFT HALF BLOCK", 221, "Half Blocks"),
    BoxChar("▍", "258D", 9613, "LEFT THREE EIGHTHS BLOCK", None, "Fractional Blocks"),
    BoxChar("▎", "258E", 9614, "LEFT ONE QUARTER BLOCK", None, "Fractional Blocks"),
    BoxChar("▏", "258F", 9615, "LEFT ONE EIGHTH BLOCK", None, "Fractional Blocks"),
    BoxChar("▐", "2590", 9616, "RIGHT HALF BLOCK", 222, "Half Blocks"),
    BoxChar("░", "2591", 9617, "LIGHT SHADE", 176, "Shade Blocks"),
    BoxChar("▒", "2592", 9618, "MEDIUM SHADE", 177, "Shade Blocks"),
    BoxChar("▓", "2593", 9619, "DARK SHADE", 178, "Shade Blocks"),
    BoxChar("▔", "2594", 9620, "UPPER ONE EIGHTH BLOCK", None, "Fractional Blocks"),
    BoxChar("▕", "2595", 9621, "RIGHT ONE EIGHTH BLOCK", None, "Fractional Blocks"),
    BoxChar("▖", "2596", 9622, "QUADRANT LOWER LEFT", None, "Quadrant Blocks"),
    BoxChar("▗", "2597", 9623, "QUADRANT LOWER RIGHT", None, "Quadrant Blocks"),
    BoxChar("▘", "2598", 9624, "QUADRANT UPPER LEFT", None, "Quadrant Blocks"),
    BoxChar("▙", "2599", 9625, "QUADRANT UPPER LEFT AND LOWER LEFT AND LOWER RIGHT", None, "Quadrant Blocks"),
    BoxChar("▚", "259A", 9626, "QUADRANT UPPER LEFT AND LOWER RIGHT", None, "Quadrant Blocks"),
    BoxChar("▛", "259B", 9627, "QUADRANT UPPER LEFT AND UPPER RIGHT AND LOWER LEFT", None, "Quadrant Blocks"),
    BoxChar("▜", "259C", 9628, "QUADRANT UPPER LEFT AND UPPER RIGHT AND LOWER RIGHT", None, "Quadrant Blocks"),
    BoxChar("▝", "259D", 9629, "QUADRANT UPPER RIGHT", None, "Quadrant Blocks"),
    BoxChar("▞", "259E", 9630, "QUADRANT UPPER RIGHT AND LOWER LEFT", None, "Quadrant Blocks"),
    BoxChar("▟", "259F", 9631, "QUADRANT UPPER RIGHT AND LOWER LEFT AND LOWER RIGHT", None, "Quadrant Blocks"),
]


# ===================================================================
# Lookup Functions
# ===================================================================

def get_all_chars():
    """Return all characters from all tables (deduplicated by unicode_hex)."""
    seen = set()
    result = []
    for char_list in [UNICODE_BOX_DRAWING, BLOCK_ELEMENTS]:
        for c in char_list:
            if c.unicode_hex not in seen:
                seen.add(c.unicode_hex)
                result.append(c)
    return result


def lookup_by_unicode(code_point: str):
    """Look up a character by Unicode code point (e.g., 'U+2554' or '2554')."""
    code_point = code_point.upper().replace("U+", "").replace("0X", "")
    for c in get_all_chars():
        if c.unicode_hex.upper() == code_point:
            return c
    return None


def lookup_by_cp437(dec: int):
    """Look up a character by CP437 decimal value."""
    for c in get_all_chars():
        if c.cp437_dec == dec:
            return c
    return None


def lookup_by_decimal(dec: int):
    """Look up a character by Unicode decimal value."""
    for c in get_all_chars():
        if c.unicode_dec == dec:
            return c
    return None


def search_by_name(query: str):
    """Search characters by name substring (case-insensitive)."""
    query = query.upper()
    return [c for c in get_all_chars() if query in c.unicode_name.upper()]


def search_by_category(category: str):
    """Search characters by category substring (case-insensitive)."""
    category = category.upper()
    return [c for c in get_all_chars() if category in c.category.upper()]


# ===================================================================
# Display / Formatting Functions
# ===================================================================

def format_char_info(c: BoxChar) -> str:
    """Format a single character's information for display."""
    cp437_info = f"  CP437: {c.cp437_dec} ({c.cp437_hex})" if c.cp437_dec else ""
    return (
        f"  Character: {c.char}\n"
        f"  Unicode:   {c.unicode_codepoint} (Dec: {c.unicode_dec})\n"
        f"  Name:      {c.unicode_name}\n"
        f"  HTML:      {c.html_dec} / {c.html_hex}\n"
        f"  Category:  {c.category}"
        f"{cp437_info}"
    )


def print_markdown_table(chars: list, title: str = "", include_cp437: bool = False):
    """Print a markdown table for a list of BoxChar objects."""
    if title:
        print(f"\n### {title}\n")

    if include_cp437:
        print("| Char | Unicode | Dec | Hex | CP437 Dec | CP437 Hex | HTML Dec | HTML Hex | Description |")
        print("|------|---------|-----|-----|-----------|-----------|----------|----------|-------------|")
        for c in chars:
            cp437_d = str(c.cp437_dec) if c.cp437_dec else "—"
            cp437_h = c.cp437_hex if c.cp437_hex else "—"
            print(f"| {c.char} | {c.unicode_codepoint} | {c.unicode_dec} | 0x{c.unicode_hex} | {cp437_d} | {cp437_h} | {c.html_dec} | {c.html_hex} | {c.unicode_name} |")
    else:
        print("| Char | Unicode | Dec | Hex | HTML Dec | HTML Hex | Description |")
        print("|------|---------|-----|-----|----------|----------|-------------|")
        for c in chars:
            print(f"| {c.char} | {c.unicode_codepoint} | {c.unicode_dec} | 0x{c.unicode_hex} | {c.html_dec} | {c.html_hex} | {c.unicode_name} |")


def print_cp437_table():
    """Print the CP437 box drawing character table sorted by CP437 decimal."""
    sorted_chars = sorted(CP437_BOX_DRAWING, key=lambda c: c.cp437_dec)
    print("\n## Code Page 437 (IBM PC) Box Drawing Characters\n")
    print("| CP437 Dec | CP437 Hex | Char | Unicode | Unicode Dec | HTML Dec | HTML Hex | Description |")
    print("|-----------|-----------|------|---------|-------------|----------|----------|-------------|")
    for c in sorted_chars:
        print(f"| {c.cp437_dec} | {c.cp437_hex} | {c.char} | {c.unicode_codepoint} | {c.unicode_dec} | {c.html_dec} | {c.html_hex} | {c.unicode_name} |")


def print_visual_examples():
    """Print visual box drawing examples."""
    print("\n## Visual Box Drawing Examples\n")

    print("### Light Single-Line Box\n")
    print("```")
    print("┌──────────────────┐")
    print("│  Light Box       │")
    print("│  (single lines)  │")
    print("├──────────────────┤")
    print("│  With divider    │")
    print("└──────────────────┘")
    print("```\n")

    print("### Heavy (Bold) Box\n")
    print("```")
    print("┏━━━━━━━━━━━━━━━━━━┓")
    print("┃  Heavy Box       ┃")
    print("┃  (bold lines)    ┃")
    print("┣━━━━━━━━━━━━━━━━━━┫")
    print("┃  With divider    ┃")
    print("┗━━━━━━━━━━━━━━━━━━┛")
    print("```\n")

    print("### Double-Line Box\n")
    print("```")
    print("╔══════════════════╗")
    print("║  Double Box      ║")
    print("║  (double lines)  ║")
    print("╠══════════════════╣")
    print("║  With divider    ║")
    print("╚══════════════════╝")
    print("```\n")

    print("### Rounded (Arc) Corners\n")
    print("```")
    print("╭──────────────────╮")
    print("│  Rounded Box     │")
    print("│  (arc corners)   │")
    print("╰──────────────────╯")
    print("```\n")

    print("### Mixed Single/Double Box\n")
    print("```")
    print("╒══════════════════╕")
    print("│  Single vertical │")
    print("│  Double horiz.   │")
    print("╘══════════════════╛")
    print("```\n")

    print("```")
    print("╓──────────────────╖")
    print("║  Double vertical ║")
    print("║  Single horiz.   ║")
    print("╙──────────────────╜")
    print("```\n")

    print("### Dashed Line Boxes\n")
    print("```")
    print("┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐      ┌┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┐")
    print("┆  Triple Dash      ┆      ┊  Quadruple Dash   ┊")
    print("└┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘      └┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┘")
    print("```\n")

    print("### Nested Boxes\n")
    print("```")
    print("╔══════════════════════════╗")
    print("║  ┌────────────────────┐  ║")
    print("║  │  ╭──────────────╮  │  ║")
    print("║  │  │  Inner Box   │  │  ║")
    print("║  │  ╰──────────────╯  │  ║")
    print("║  └────────────────────┘  ║")
    print("╚══════════════════════════╝")
    print("```\n")

    print("### Table Layout\n")
    print("```")
    print("┌─────────┬─────────┬─────────┐")
    print("│ Header1 │ Header2 │ Header3 │")
    print("╞═════════╪═════════╪═════════╡")
    print("│ Cell 1  │ Cell 2  │ Cell 3  │")
    print("├─────────┼─────────┼─────────┤")
    print("│ Cell 4  │ Cell 5  │ Cell 6  │")
    print("└─────────┴─────────┴─────────┘")
    print("```\n")

    print("### Block Elements - Progress Bar\n")
    print("```")
    print("Loading: █████████████▓▒░░░░░░░░░░ 52%")
    print("Loading: ████████████████████████░░ 92%")
    print("Loading: ██████████████████████████ 100%")
    print("```\n")

    print("### Block Elements - Bar Chart\n")
    print("```")
    print("  100% ┤ █")
    print("   80% ┤ █ █")
    print("   60% ┤ █ █   █")
    print("   40% ┤ █ █ █ █")
    print("   20% ┤ █ █ █ █ █")
    print("       └─┴─┴─┴─┴─┘")
    print("         A B C D E")
    print("```\n")

    print("### Shade Gradient\n")
    print("```")
    print("░░░░▒▒▒▒▓▓▓▓████")
    print("Light → Medium → Dark → Full")
    print("```\n")

    print("### Tree Diagram\n")
    print("```")
    print("project/")
    print("├── src/")
    print("│   ├── main.py")
    print("│   ├── utils/")
    print("│   │   ├── helpers.py")
    print("│   │   └── constants.py")
    print("│   └── tests/")
    print("│       └── test_main.py")
    print("├── docs/")
    print("│   └── README.md")
    print("└── setup.py")
    print("```\n")


def print_full_markdown_document():
    """Output the complete markdown document suitable for BOX_DRAWING_ASCII.md."""
    # This generates the complete document content to stdout
    # Redirect with: python box_drawing_lookup.py --markdown-full > output.md

    print("# Box Drawing, Block Element, and Text Characters")
    print()
    print("Comprehensive reference for Unicode Box Drawing characters (U+2500–U+257F),")
    print("Block Elements (U+2580–U+259F), and their Code Page 437 (IBM PC / DOS) equivalents.")
    print()
    print("**Generated by**: `util/box_drawing_lookup.py`")
    print()
    print("---")
    print()
    print("## Table of Contents")
    print()
    print("1. [Background and Encoding Notes](#background-and-encoding-notes)")
    print("2. [Code Page 437 Box Drawing Characters](#code-page-437-ibm-pc-box-drawing-characters)")
    print("3. [Unicode Box Drawing Block (U+2500–U+257F)](#unicode-box-drawing-block-u2500u257f)")
    print("4. [Unicode Block Elements (U+2580–U+259F)](#unicode-block-elements-u2580u259f)")
    print("5. [Visual Examples](#visual-examples)")
    print("6. [Utility Scripts](#utility-scripts)")
    print()
    print("---")
    print()

    # Background section
    print("## Background and Encoding Notes")
    print()
    print("### Code Page 437 (CP437)")
    print()
    print("Code Page 437 is the original character set of the IBM PC (1981), also known as")
    print("DOS Latin US or OEM-US. It extends ASCII (0–127) with 128 additional characters")
    print("(128–255) including box drawing characters, block elements, mathematical symbols,")
    print("and accented letters. The box drawing characters occupy CP437 positions 176–223")
    print("and were widely used in DOS text-mode user interfaces (TUIs).")
    print()
    print("### Unicode")
    print()
    print("Unicode consolidates box drawing characters into dedicated blocks:")
    print()
    print("| Block | Range | Count | Description |")
    print("|-------|-------|-------|-------------|")
    print("| Box Drawing | U+2500–U+257F | 128 | Lines, corners, T-junctions, crosses |")
    print("| Block Elements | U+2580–U+259F | 32 | Full/half/fractional blocks, shades, quadrants |")
    print()
    print("### Encoding Cross-Reference")
    print()
    print("Not all box drawing characters have CP437 equivalents. CP437 includes 48 box drawing")
    print("characters and a few block/shade characters (positions 176–223). Unicode provides the")
    print("full set of 128 box drawing characters plus 32 block elements, including many")
    print("light/heavy mixed variants and arc corners that CP437 lacks.")
    print()
    print("### Character Weight Terminology")
    print()
    print("| Term | Appearance | Example |")
    print("|------|------------|---------|")
    print("| **Light** | Single thin line | ─ │ ┌ ┐ └ ┘ |")
    print("| **Heavy** | Single thick/bold line | ━ ┃ ┏ ┓ ┗ ┛ |")
    print("| **Double** | Two parallel thin lines | ═ ║ ╔ ╗ ╚ ╝ |")
    print("| **Dashed** | Broken line segments | ┄ ┅ ┆ ┇ ┈ ┉ |")
    print("| **Arc** | Rounded/curved corners | ╭ ╮ ╯ ╰ |")
    print()
    print("### HTML Entity Formats")
    print()
    print("Box drawing characters can be inserted in HTML using:")
    print()
    print("| Format | Example (╔) | Description |")
    print("|--------|-------------|-------------|")
    print("| Hex entity | `&#x2554;` | Unicode hex code point |")
    print("| Dec entity | `&#9556;` | Unicode decimal value |")
    print("| Direct | `╔` | UTF-8 encoded character (preferred) |")
    print()
    print("---")
    print()

    # CP437 table
    print_cp437_table()
    print()
    print("---")
    print()

    # Unicode Box Drawing by category
    print("## Unicode Box Drawing Block (U+2500–U+257F)")
    print()
    print("All 128 characters in the Unicode Box Drawing block, organized by category.")
    print()

    categories = [
        ("Lines (Light and Heavy)", "Lines"),
        ("Dashed Lines (Triple, Quadruple, Double Dash)", "Dashed Lines"),
        ("Corners (Light, Heavy, and Mixed)", "Corners"),
        ("T-Junctions (Light, Heavy, and Mixed)", "T-Junctions"),
        ("Crosses (Light, Heavy, and Mixed)", "Crosses"),
        ("Double Lines", "Double Lines"),
        ("Double Corners", "Double Corners"),
        ("Single/Double Transition Corners", "Single/Double Corners"),
        ("Double T-Junctions", "Double T-Junctions"),
        ("Single/Double Transition T-Junctions", "Single/Double T-Junctions"),
        ("Single/Double Transition Crosses", "Single/Double Crosses"),
        ("Double Crosses", "Double Crosses"),
        ("Arc Corners (Rounded)", "Arc Corners"),
        ("Diagonal Lines", "Diagonals"),
        ("Half-Line Segments", "Half Lines"),
        ("Weight Transition Lines", "Weight Transitions"),
    ]

    for title, cat in categories:
        chars = [c for c in UNICODE_BOX_DRAWING if c.category == cat]
        if chars:
            print_markdown_table(chars, title, include_cp437=True)
            print()

    print("---")
    print()

    # Block Elements
    print("## Unicode Block Elements (U+2580–U+259F)")
    print()
    print("All 32 characters in the Unicode Block Elements block.")
    print()

    block_categories = [
        ("Half Blocks", "Half Blocks"),
        ("Full Block", "Full Block"),
        ("Fractional Blocks (Lower and Left Series)", "Fractional Blocks"),
        ("Shade Blocks", "Shade Blocks"),
        ("Quadrant Blocks", "Quadrant Blocks"),
    ]

    for title, cat in block_categories:
        chars = [c for c in BLOCK_ELEMENTS if c.category == cat]
        if chars:
            print_markdown_table(chars, title, include_cp437=True)
            print()

    print("---")
    print()

    # Visual Examples
    print_visual_examples()

    print("---")
    print()

    # Complete Sequential Reference
    print("## Complete Sequential Reference (U+2500–U+257F)")
    print()
    print("All 128 Box Drawing characters in code point order for verification.")
    print()
    print("| # | Char | Unicode | Dec | Hex | CP437 | Name |")
    print("|---|------|---------|-----|-----|-------|------|")
    for i, c in enumerate(UNICODE_BOX_DRAWING, 1):
        cp437 = str(c.cp437_dec) if c.cp437_dec else "—"
        # Shorten name by removing "BOX DRAWINGS " prefix
        short_name = c.unicode_name.replace("BOX DRAWINGS ", "")
        print(f"| {i} | {c.char} | {c.unicode_codepoint} | {c.unicode_dec} | 0x{c.unicode_hex} | {cp437} | {short_name} |")
    print()

    print("---")
    print()

    # Summary Statistics
    print("## Summary Statistics")
    print()
    print("| Block | Characters | With CP437 Equivalent |")
    print("|-------|------------|-----------------------|")
    bd_cp437 = len([c for c in UNICODE_BOX_DRAWING if c.cp437_dec is not None])
    be_cp437 = len([c for c in BLOCK_ELEMENTS if c.cp437_dec is not None])
    print(f"| Box Drawing (U+2500–U+257F) | {len(UNICODE_BOX_DRAWING)} | {bd_cp437} |")
    print(f"| Block Elements (U+2580–U+259F) | {len(BLOCK_ELEMENTS)} | {be_cp437} |")
    print(f"| **Total** | **{len(UNICODE_BOX_DRAWING) + len(BLOCK_ELEMENTS)}** | **{bd_cp437 + be_cp437}** |")
    print()

    # Category breakdown
    all_chars = get_all_chars()
    cats = {}
    for c in all_chars:
        cats[c.category] = cats.get(c.category, 0) + 1
    print("### Characters by Category")
    print()
    print("| Category | Count |")
    print("|----------|-------|")
    for cat in sorted(cats.keys()):
        print(f"| {cat} | {cats[cat]} |")
    print()

    print("---")
    print()

    # Utility Scripts
    print("## Utility Scripts")
    print()
    print("### `util/box_drawing_lookup.py`")
    print()
    print("Python utility for looking up and generating box drawing character tables.")
    print()
    print("```bash")
    print("# Show all tables")
    print("python util/box_drawing_lookup.py")
    print()
    print("# Search by description keyword")
    print("python util/box_drawing_lookup.py --search \"corner\"")
    print("python util/box_drawing_lookup.py --search \"double\"")
    print("python util/box_drawing_lookup.py --search \"arc\"")
    print()
    print("# Show CP437 mapping table")
    print("python util/box_drawing_lookup.py --cp437")
    print()
    print("# Show full Unicode box drawing table")
    print("python util/box_drawing_lookup.py --unicode")
    print()
    print("# Show block elements table")
    print("python util/box_drawing_lookup.py --blocks")
    print()
    print("# Show visual box drawing examples")
    print("python util/box_drawing_lookup.py --examples")
    print()
    print("# Look up a specific character")
    print("python util/box_drawing_lookup.py --lookup U+2554    # By Unicode code point")
    print("python util/box_drawing_lookup.py --lookup 201       # By CP437 decimal")
    print()
    print("# Generate markdown tables")
    print("python util/box_drawing_lookup.py --markdown")
    print()
    print("# Generate the complete markdown document")
    print("python util/box_drawing_lookup.py --markdown-full > notes/BOX_DRAWING_ASCII.md")
    print("```")
    print()
    print("---")
    print()
    print("*Document generated by `util/box_drawing_lookup.py`*")


# ===================================================================
# CLI Entry Point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Box Drawing and Block Element Character Lookup Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         Show summary of all character tables
  %(prog)s --search "corner"       Search by description keyword
  %(prog)s --search "double"       Find all double-line characters
  %(prog)s --cp437                  Show CP437 box drawing table
  %(prog)s --unicode                Show full Unicode box drawing table
  %(prog)s --blocks                 Show block elements table
  %(prog)s --examples               Show visual box drawing examples
  %(prog)s --lookup U+2554          Look up by Unicode code point
  %(prog)s --lookup 201             Look up by CP437 decimal value
  %(prog)s --markdown               Output all tables in markdown
  %(prog)s --markdown-full          Output the complete reference document
        """
    )

    parser.add_argument("--search", "-s", type=str, help="Search characters by description keyword")
    parser.add_argument("--lookup", "-l", type=str, help="Look up a specific character (U+XXXX or CP437 decimal)")
    parser.add_argument("--cp437", action="store_true", help="Show CP437 box drawing character table")
    parser.add_argument("--unicode", "-u", action="store_true", help="Show full Unicode box drawing table")
    parser.add_argument("--blocks", "-b", action="store_true", help="Show block elements table")
    parser.add_argument("--examples", "-e", action="store_true", help="Show visual box drawing examples")
    parser.add_argument("--markdown", "-m", action="store_true", help="Output all tables in markdown format")
    parser.add_argument("--markdown-full", action="store_true", help="Output the complete markdown document")
    parser.add_argument("--category", "-c", type=str, help="Filter by category")

    args = parser.parse_args()

    # If no arguments, show summary
    if len(sys.argv) == 1:
        print("Box Drawing and Block Element Character Lookup Utility")
        print("=" * 55)
        print()
        print(f"  Box Drawing characters (U+2500–U+257F): {len(UNICODE_BOX_DRAWING)}")
        print(f"  Block Elements (U+2580–U+259F):          {len(BLOCK_ELEMENTS)}")
        print(f"  CP437 mapped characters:                  {len(CP437_BOX_DRAWING)}")
        print(f"  Total unique characters:                  {len(get_all_chars())}")
        print()
        print("Use --help for full usage information.")
        print()

        # Show categories
        all_chars = get_all_chars()
        cats = {}
        for c in all_chars:
            cats[c.category] = cats.get(c.category, 0) + 1
        print("Categories:")
        for cat in sorted(cats.keys()):
            # Show sample characters
            samples = [c.char for c in all_chars if c.category == cat][:8]
            print(f"  {cat:40s} ({cats[cat]:3d})  {''.join(samples)}")
        return

    if args.markdown_full:
        print_full_markdown_document()
        return

    if args.lookup:
        query = args.lookup.strip()
        result = None

        # Try Unicode code point first
        if query.upper().startswith("U+") or query.upper().startswith("0X"):
            result = lookup_by_unicode(query)
        elif query.isdigit():
            dec = int(query)
            if dec < 256:
                # Try CP437 first
                result = lookup_by_cp437(dec)
                if result:
                    print(f"Found by CP437 decimal {dec}:")
            if not result:
                result = lookup_by_decimal(dec)
                if result:
                    print(f"Found by Unicode decimal {dec}:")

        if result:
            print(format_char_info(result))
        else:
            # Try as hex without prefix
            result = lookup_by_unicode(query)
            if result:
                print(f"Found by Unicode hex {query}:")
                print(format_char_info(result))
            else:
                print(f"Character not found: {query}")
        return

    if args.search:
        results = search_by_name(args.search)
        if results:
            print(f"\nFound {len(results)} characters matching '{args.search}':\n")
            print_markdown_table(results, include_cp437=True)
        else:
            print(f"No characters found matching '{args.search}'")
        return

    if args.category:
        results = search_by_category(args.category)
        if results:
            print(f"\nFound {len(results)} characters in category '{args.category}':\n")
            print_markdown_table(results, include_cp437=True)
        else:
            print(f"No characters found in category '{args.category}'")
        return

    if args.cp437:
        print_cp437_table()
        return

    if args.unicode:
        print("\n## Unicode Box Drawing Block (U+2500–U+257F)\n")
        print_markdown_table(UNICODE_BOX_DRAWING, include_cp437=True)
        return

    if args.blocks:
        print("\n## Unicode Block Elements (U+2580–U+259F)\n")
        print_markdown_table(BLOCK_ELEMENTS, include_cp437=True)
        return

    if args.examples:
        print_visual_examples()
        return

    if args.markdown:
        print_cp437_table()
        print()
        print("\n## Unicode Box Drawing Block (U+2500–U+257F)\n")
        print_markdown_table(UNICODE_BOX_DRAWING, include_cp437=True)
        print()
        print("\n## Unicode Block Elements (U+2580–U+259F)\n")
        print_markdown_table(BLOCK_ELEMENTS, include_cp437=True)
        return


if __name__ == "__main__":
    main()
