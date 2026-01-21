"""
Entity Extraction Service
Extracts vehicle make/model/year and DTC codes from text.
"""
import re
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VehicleInfo(BaseModel):
    """Extracted vehicle information."""
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    raw_text: Optional[str] = None


class DTCCode(BaseModel):
    """Diagnostic Trouble Code information."""
    code: str
    category: str
    description: str
    system: str


# Common vehicle makes (expandable)
VEHICLE_MAKES = {
    # International
    "toyota", "honda", "ford", "chevrolet", "chevy", "nissan", "hyundai",
    "kia", "bmw", "mercedes", "audi", "volkswagen", "vw", "mazda", "subaru",
    "tesla", "jeep", "dodge", "ram", "gmc", "lexus", "acura", "infiniti",
    # Indian Cars
    "maruti", "suzuki", "tata", "mahindra", "renault", "skoda", "mg",
    "citroen", "kia", "toyota", "honda", "hyundai", "volkswagen",
    # Indian Two-Wheelers
    "hero", "bajaj", "tvs", "royal enfield", "honda", "yamaha", "suzuki",
    "ktm", "jawa", "ola", "ather", "revolt", "activa", "splendor", "pulsar"
}

# Common models
VEHICLE_MODELS = {
    # Japanese
    "camry", "corolla", "rav4", "highlander", "prius",
    "civic", "accord", "crv", "pilot", "odyssey", "city", "amaze",
    "altima", "sentra", "rogue", "pathfinder",
    # Korean
    "elantra", "sonata", "tucson", "santa fe", "i20", "i10", "venue", "creta", "verna",
    "seltos", "sonet", "carnival", "carens",
    # Indian Cars - Maruti
    "swift", "baleno", "brezza", "dzire", "ertiga", "alto", "wagon r", "celerio",
    "ignis", "xl6", "ciaz", "s-presso", "fronx", "jimny", "grand vitara", "invicto",
    # Indian Cars - Tata
    "nexon", "harrier", "safari", "punch", "altroz", "tiago", "tigor", "curvv",
    # Indian Cars - Mahindra
    "scorpio", "thar", "xuv700", "bolero", "xuv300", "xuv400", "marazzo", "xylo",
    # Indian Two-Wheelers
    "splendor", "passion", "glamour", "xpulse", "xtreme",  # Hero
    "pulsar", "dominar", "avenger", "platina", "ct100",  # Bajaj
    "apache", "jupiter", "ntorq", "raider",  # TVS
    "classic 350", "bullet", "himalayan", "meteor", "hunter",  # Royal Enfield
    "activa", "shine", "unicorn", "hornet", "cb350",  # Honda
    "fz", "r15", "mt15", "fascino", "ray",  # Yamaha
    # American
    "f150", "mustang", "explorer", "escape",
    "silverado", "equinox", "tahoe", "malibu",
    # German
    "jetta", "passat", "tiguan", "golf", "polo", "virtus", "taigun",
    "3 series", "5 series", "x3", "x5",
    "a4", "a6", "q5", "q7",
    "c class", "e class", "glc", "gle"
}

# DTC Code patterns and meanings
DTC_PATTERNS = {
    "P0": "Powertrain - Generic",
    "P1": "Powertrain - Manufacturer Specific",
    "P2": "Powertrain - Generic",
    "P3": "Powertrain - Generic/Manufacturer",
    "B0": "Body - Generic",
    "B1": "Body - Manufacturer Specific",
    "C0": "Chassis - Generic",
    "C1": "Chassis - Manufacturer Specific",
    "U0": "Network - Generic",
    "U1": "Network - Manufacturer Specific"
}

# Common DTC codes with descriptions
DTC_DATABASE = {
    # Engine/Fuel
    "P0300": ("Random/Multiple Cylinder Misfire", "POWERTRAIN"),
    "P0301": ("Cylinder 1 Misfire", "POWERTRAIN"),
    "P0302": ("Cylinder 2 Misfire", "POWERTRAIN"),
    "P0303": ("Cylinder 3 Misfire", "POWERTRAIN"),
    "P0304": ("Cylinder 4 Misfire", "POWERTRAIN"),
    "P0171": ("System Too Lean (Bank 1)", "POWERTRAIN"),
    "P0172": ("System Too Rich (Bank 1)", "POWERTRAIN"),
    "P0420": ("Catalyst System Efficiency Below Threshold", "EXHAUST"),
    "P0455": ("EVAP System Large Leak Detected", "POWERTRAIN"),
    "P0442": ("EVAP System Small Leak Detected", "POWERTRAIN"),
    
    # Sensors
    "P0101": ("MAF Sensor Circuit Range/Performance", "POWERTRAIN"),
    "P0113": ("Intake Air Temperature Sensor High", "POWERTRAIN"),
    "P0117": ("Engine Coolant Temperature Sensor Low", "POWERTRAIN"),
    "P0128": ("Coolant Thermostat Below Thermostat Regulating Temperature", "POWERTRAIN"),
    "P0131": ("O2 Sensor Circuit Low Voltage (Bank 1, Sensor 1)", "POWERTRAIN"),
    "P0141": ("O2 Sensor Heater Circuit Malfunction (Bank 1, Sensor 2)", "POWERTRAIN"),
    
    # Transmission
    "P0700": ("Transmission Control System Malfunction", "POWERTRAIN"),
    "P0715": ("Input/Turbine Speed Sensor Circuit Malfunction", "POWERTRAIN"),
    "P0720": ("Output Speed Sensor Circuit Malfunction", "POWERTRAIN"),
    "P0730": ("Incorrect Gear Ratio", "POWERTRAIN"),
    
    # ABS/Chassis
    "C0035": ("Left Front Wheel Speed Sensor Circuit", "BRAKES"),
    "C0040": ("Right Front Wheel Speed Sensor Circuit", "BRAKES"),
    "C0045": ("Left Rear Wheel Speed Sensor Circuit", "BRAKES"),
    "C0050": ("Right Rear Wheel Speed Sensor Circuit", "BRAKES"),
    
    # Electrical
    "P0562": ("System Voltage Low", "ELECTRICAL"),
    "P0563": ("System Voltage High", "ELECTRICAL"),
}


class EntityExtractor:
    """Extracts entities like vehicle info and DTC codes from text."""
    
    def __init__(self):
        self.year_pattern = re.compile(r'\b(19[89]\d|20[0-2]\d)\b')
        self.dtc_pattern = re.compile(r'\b([PBCU][01][0-9A-F]{3})\b', re.IGNORECASE)
    
    def extract_vehicle_info(self, text: str) -> VehicleInfo:
        """Extract vehicle make, model, year from text."""
        text_lower = text.lower()
        
        # Find year
        year_match = self.year_pattern.search(text)
        year = int(year_match.group(1)) if year_match else None
        
        # Find make
        make = None
        for m in VEHICLE_MAKES:
            if m in text_lower:
                make = m.title()
                break
        
        # Find model
        model = None
        for mod in VEHICLE_MODELS:
            if mod in text_lower:
                model = mod.title()
                break
        
        return VehicleInfo(
            make=make,
            model=model,
            year=year,
            raw_text=text if any([make, model, year]) else None
        )
    
    def extract_dtc_codes(self, text: str) -> List[DTCCode]:
        """Extract and decode DTC codes from text."""
        codes = []
        matches = self.dtc_pattern.findall(text)
        
        for match in matches:
            code = match.upper()
            
            # Look up in database
            if code in DTC_DATABASE:
                desc, system = DTC_DATABASE[code]
            else:
                # Generate generic description
                prefix = code[:2]
                category = DTC_PATTERNS.get(prefix, "Unknown")
                desc = f"{category} code"
                system = "UNKNOWN"
            
            codes.append(DTCCode(
                code=code,
                category=code[0],  # P, B, C, or U
                description=desc if code in DTC_DATABASE else f"Code {code} - {category}",
                system=system
            ))
        
        return codes
    
    def extract_all(self, text: str) -> Dict:
        """Extract all entities from text."""
        return {
            "vehicle": self.extract_vehicle_info(text),
            "dtc_codes": self.extract_dtc_codes(text)
        }


# Singleton
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor
