import pandas as pd

class FeatureEngineering:
    def __init__(self):
        self.original_categorical_features = [
            'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families'
        ]

        self.original_numerical_features = [
            'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 
            'land_surface_condition', 'foundation_type', 'roof_type', 
            'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
            'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 
            'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 
            'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 
            'has_superstructure_rc_engineered', 'has_superstructure_other', 'legal_ownership_status', 
            'has_secondary_use', 'has_secondary_use_agriculture', 'has_secondary_use_hotel',
            'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 
            'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 
            'has_secondary_use_use_police', 'has_secondary_use_other'
        ]

    def add_building_material_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Adds building material-related features to the DataFrame."""
        df = df.copy()

        # Sticking material
        df["sticking_material"] = df[[
            'has_superstructure_mud_mortar_stone',
            'has_superstructure_mud_mortar_brick',
            'has_superstructure_cement_mortar_stone',
            'has_superstructure_cement_mortar_brick']].idxmax(axis=1)

        df["sticking_material"] = df["sticking_material"].map({
            'has_superstructure_mud_mortar_stone': 'mud',
            'has_superstructure_mud_mortar_brick': 'mud',
            'has_superstructure_cement_mortar_stone': 'cement',
            'has_superstructure_cement_mortar_brick': 'cement'
        }).fillna("none")

        # Building material
        df["building_material"] = df[[
            'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_stone',
            'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo'
        ]].idxmax(axis=1)

        df["building_material"] = df["building_material"].map({
            'has_superstructure_adobe_mud': 'adobe',
            'has_superstructure_mud_mortar_stone': 'stone',
            'has_superstructure_stone_flag': 'stone',
            'has_superstructure_mud_mortar_brick': 'brick',
            'has_superstructure_cement_mortar_stone': 'stone',
            'has_superstructure_cement_mortar_brick': 'brick',
            'has_superstructure_timber': 'wood',
            'has_superstructure_bamboo': 'wood'
        }).fillna("other")

        # Is Concrete
        df["is_concrete"] = df[['has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered']].max(axis=1)
        df["is_concrete"] = df["is_concrete"].apply(lambda x: "True" if x > 0 else "False")

        return df

    def add_building_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Adds type of building-related features to the DataFrame."""
        df = df.copy()

        df["type_of_building"] = df[[
            'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental', 
            'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry',
            'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police', 
            'has_secondary_use_other', 'has_secondary_use']].idxmax(axis=1)

        df["type_of_building"] = df["type_of_building"].map({
            'has_secondary_use_agriculture': "agriculture",
            'has_secondary_use_hotel': "institutional",
            'has_secondary_use_rental': "other", 
            'has_secondary_use_institution': "institutional",
            'has_secondary_use_school': "institutional", 
            'has_secondary_use_industry': "industrial",
            'has_secondary_use_health_post': "other", 
            'has_secondary_use_gov_office': "institutional",
            'has_secondary_use_use_police': "institutional", 
            'has_secondary_use_other': "other",
            'has_secondary_use': "other"
        }).fillna("other")
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Applies all feature engineering transformations."""
        df = self.add_building_material_features(df)
        df = self.add_building_type_features(df)
        return df
