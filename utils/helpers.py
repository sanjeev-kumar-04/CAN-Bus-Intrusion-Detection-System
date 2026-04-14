import json

def save_sample_data(df, filename='sample_data.json'):
    sample_data = {
        'normal': df[df['Attack_Type'] == 'Normal'].sample(n=100).to_dict('records'),
        'dos': df[df['Attack_Type'] == 'DoS'].sample(n=100).to_dict('records'),
        'fuzzy': df[df['Attack_Type'] == 'Fuzzy'].sample(n=100).to_dict('records'),
        'gear': df[df['Attack_Type'] == 'Gear'].sample(n=100).to_dict('records'),
        'rpm': df[df['Attack_Type'] == 'RPM'].sample(n=100).to_dict('records')
    }

    with open(filename, 'w') as f:
        json.dump(sample_data, f)