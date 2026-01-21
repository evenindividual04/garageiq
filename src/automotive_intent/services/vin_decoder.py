"""
VIN Decoder Service
Extracts vehicle metadata from VIN for variant-specific RAG filtering.
"""
import re
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VehicleInfo:
    """Decoded vehicle information from VIN."""
    vin: str
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None
    trim: Optional[str] = None
    body_style: Optional[str] = None
    drive_type: Optional[str] = None  # FWD, RWD, AWD
    fuel_type: Optional[str] = None  # Petrol, Diesel, CNG, Electric
    
    def get_filter_tags(self) -> list:
        """Return list of tags for RAG filtering."""
        tags = []
        if self.make:
            tags.append(f"make:{self.make.lower()}")
        if self.model:
            tags.append(f"model:{self.model.lower()}")
        if self.engine:
            tags.append(f"engine:{self.engine.lower()}")
        if self.fuel_type:
            tags.append(f"fuel:{self.fuel_type.lower()}")
        if self.year:
            # Generation grouping
            if self.year >= 2020:
                tags.append("gen:current")
            elif self.year >= 2015:
                tags.append("gen:previous")
            else:
                tags.append("gen:legacy")
        return tags


# VIN character to year mapping (position 10)
YEAR_CODES = {
    'A': 2010, 'B': 2011, 'C': 2012, 'D': 2013, 'E': 2014,
    'F': 2015, 'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019,
    'L': 2020, 'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024,
    'S': 2025, 'T': 2026, 'V': 2027, 'W': 2028, 'X': 2029,
    'Y': 2030,
    '1': 2001, '2': 2002, '3': 2003, '4': 2004, '5': 2005,
    '6': 2006, '7': 2007, '8': 2008, '9': 2009,
}

# WMI (World Manufacturer Identifier) - first 3 chars
WMI_TO_MAKE = {
    # Indian Manufacturers
    "MA3": "Suzuki",
    "MA1": "Mahindra",
    "MAT": "Tata",
    "MBH": "Honda",
    "MAK": "Toyota",
    "MAL": "Hyundai",
    "MBJ": "Maruti Suzuki",
    
    # Global
    "1G1": "Chevrolet",
    "1HG": "Honda",
    "1FA": "Ford",
    "2HG": "Honda",
    "3VW": "Volkswagen",
    "5YJ": "Tesla",
    "JHM": "Honda",
    "JT2": "Toyota",
    "KM8": "Hyundai",
    "KNA": "Kia",
    "WAU": "Audi",
    "WBA": "BMW",
    "WDB": "Mercedes-Benz",
    "WF0": "Ford",
    "WVW": "Volkswagen",
}

# Indian vehicle registration pattern
INDIAN_REG_PATTERN = re.compile(
    r'^([A-Z]{2})[\s-]?(\d{1,2})[\s-]?([A-Z]{1,3})[\s-]?(\d{1,4})$',
    re.IGNORECASE
)


class VINDecoder:
    """Decodes VIN to extract vehicle metadata."""
    
    def decode(self, vin_or_reg: str) -> Optional[VehicleInfo]:
        """
        Decode VIN or Indian registration number.
        
        Args:
            vin_or_reg: 17-char VIN or Indian registration (e.g., MH12AB1234)
            
        Returns:
            VehicleInfo if valid, None otherwise
        """
        vin_or_reg = vin_or_reg.strip().upper().replace(" ", "").replace("-", "")
        
        # Check if Indian registration
        if INDIAN_REG_PATTERN.match(vin_or_reg):
            return self._decode_indian_reg(vin_or_reg)
        
        # Standard VIN
        if len(vin_or_reg) == 17:
            return self._decode_vin(vin_or_reg)
        
        logger.warning(f"Invalid VIN/Registration: {vin_or_reg}")
        return None
    
    def _decode_vin(self, vin: str) -> VehicleInfo:
        """Decode standard 17-character VIN."""
        info = VehicleInfo(vin=vin)
        
        # Position 1-3: WMI (Manufacturer)
        wmi = vin[:3]
        info.make = WMI_TO_MAKE.get(wmi)
        
        # Position 10: Model Year
        year_code = vin[9]
        info.year = YEAR_CODES.get(year_code)
        
        # Position 4-8: VDS (Vehicle Descriptor Section)
        # This varies by manufacturer, simplified mapping:
        vds = vin[3:8]
        
        # Position 8: Engine/Check digit
        engine_code = vin[7]
        info.engine = self._guess_engine(engine_code, info.make)
        
        logger.info(f"Decoded VIN: {info.make} {info.year}")
        return info
    
    def _decode_indian_reg(self, reg: str) -> VehicleInfo:
        """Decode Indian vehicle registration."""
        # Can't get much from registration alone
        # but we can identify state
        info = VehicleInfo(vin=reg)
        
        state_code = reg[:2]
        indian_states = {
            "MH": "Maharashtra", "DL": "Delhi", "KA": "Karnataka",
            "TN": "Tamil Nadu", "UP": "Uttar Pradesh", "GJ": "Gujarat",
            "RJ": "Rajasthan", "WB": "West Bengal", "AP": "Andhra Pradesh",
        }
        # Registration doesn't tell us make/model
        # But we set fuel type heuristic for common patterns
        
        logger.info(f"Decoded Indian registration from {indian_states.get(state_code, 'Unknown')}")
        return info
    
    def _guess_engine(self, code: str, make: str) -> Optional[str]:
        """Heuristic engine guess based on code and make."""
        # This is manufacturer-specific
        if make == "Honda":
            return {"1": "1.5L", "2": "2.0L", "3": "2.4L"}.get(code)
        elif make == "Toyota":
            return {"A": "2.0L", "B": "2.5L", "C": "3.0L"}.get(code)
        elif make in ["Maruti Suzuki", "Suzuki"]:
            return {"K": "1.2L K-Series", "D": "1.3L Diesel"}.get(code)
        return None


# Singleton
_decoder = None

def get_vin_decoder() -> VINDecoder:
    global _decoder
    if _decoder is None:
        _decoder = VINDecoder()
    return _decoder
