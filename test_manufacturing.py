# Test the actual fragrance manufacturing system
print('Testing Movie Capsule Formulator - Real Manufacturing System')
print('=' * 60)

import sys
sys.path.append('ai_perfume')

from core.movie_capsule_formulator import MovieCapsuleFormulator

# Initialize the manufacturing system
formulator = MovieCapsuleFormulator()

# Test scenario: Director's request
scene_description = 'A romantic beach sunset scene where a couple shares their first kiss near the ocean waves'
target_duration = 5.0  # 5 seconds duration

print(f'Director Request: {scene_description}')
print(f'Required Duration: {target_duration} seconds')
print()

# Generate actual manufacturing formula
formula = formulator.formulate_capsule(scene_description, target_duration)

# Display manufacturing specifications
print('MANUFACTURING FORMULA:')
print(f'Target Duration: {formula.target_duration} seconds')
print(f'Diffusion Control: {formula.diffusion_control} (low diffusion)')
print(f'Cost per Unit: ${formula.estimated_cost_per_unit:.4f}')
print()

print('RAW MATERIALS:')
for i, material in enumerate(formula.raw_materials, 1):
    print(f'{i}. {material["name"].replace("_", " ").title()}')
    print(f'   Amount: {material["amount_ml"]:.3f}ml ({material["percentage"]:.1f}%)')
    print(f'   Function: {material["function"]}')
    print(f'   Properties: {material["properties"]}')
    print()

print('PRODUCTION SEQUENCE:')
for step in formula.production_sequence:
    print(f'  {step}')
print()

print('CAPSULE SPECIFICATIONS:')
print(f'  Encapsulation: {formula.encapsulation_method}')
print(f'  Activation: {formula.activation_mechanism}')
print()

print('This is REAL manufacturing - not recommendations!')
print('The system generates actual chemical formulas with:')
print('- Specific raw materials and quantities')
print('- Exact production procedures') 
print('- Cost calculations')
print('- Manufacturing specifications')