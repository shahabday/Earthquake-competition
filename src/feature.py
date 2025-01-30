import pandas as pd

class FeatureEngineering:
    def __init__(self):
        self.original_numerical_features  = [
            'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families'
        ]

        self.original_categorical_features= [
            'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 
            'land_surface_condition', 'foundation_type', 'roof_type', 
            'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
            'legal_ownership_status'
        ]

        self.has_flags = [
            'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 
            'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 
            'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 
            'has_superstructure_rc_engineered', 'has_superstructure_other', 'has_secondary_use', 
            'has_secondary_use_agriculture', 'has_secondary_use_hotel',
            'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 
            'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 
            'has_secondary_use_use_police', 'has_secondary_use_other'
        ]

    def add_building_material_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Adds building material-related features to the DataFrame."""
        df = df.copy()

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

    def transform(self, df: pd.DataFrame, scenario: int) -> (pd.DataFrame, list, list):
        """
        Applies feature engineering transformations and returns numerical and categorical feature lists based on scenario.

        Parameters:
        - df: Input DataFrame
        - scenario: Experiment scenario (1 to 4)

        Returns:
        - Transformed DataFrame
        - List of numerical features
        - List of categorical features
        """
        df = df.copy()
        numerical_features = self.original_numerical_features.copy()
        categorical_features = self.original_categorical_features.copy()

        if scenario in [2, 3]:
            df = self.add_building_material_features(df)
            df = self.add_building_type_features(df)
            categorical_features.extend(["sticking_material", "building_material", "is_concrete", "type_of_building"])

        if scenario in [3, 4]:  # Remove all `has_` features
            df.drop(columns=self.has_flags, inplace=True, errors="ignore")

        return df, numerical_features, categorical_features


if __name__ == "__main__": 
    # Load dataset (replace with actual data)
    df = pd.read_csv("your_dataset.csv") 

    feature_engineering = FeatureEngineering()

    # Scenario 1: All Features (Original Features)
    df_scenario1, num_features1, cat_features1 = feature_engineering.transform(df, scenario=1)

    # Scenario 2: All Features + New Features
    df_scenario2, num_features2, cat_features2 = feature_engineering.transform(df, scenario=2)

    # Scenario 3: All Features + New Features - has_flags
    df_scenario3, num_features3, cat_features3 = feature_engineering.transform(df, scenario=3)

    # Scenario 4: All Features - has_flags
    df_scenario4, num_features4, cat_features4 = feature_engineering.transform(df, scenario=4)

    print(f"Scenario 1: {len(num_features1)} numerical, {len(cat_features1)} categorical")
    print(f"Scenario 2: {len(num_features2)} numerical, {len(cat_features2)} categorical")
    print(f"Scenario 3: {len(num_features3)} numerical, {len(cat_features3)} categorical")
    print(f"Scenario 4: {len(num_features4)} numerical, {len(cat_features4)} categorical")
