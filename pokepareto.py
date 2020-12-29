import pandas as pd
import numpy as np
import pokebase as pb
from itables import init_notebook_mode
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.error import HTTPError
from IPython.display import Image
import umap
import pickle 
import logging

def load_pickle(filename):
    with open(filename, 'rb') as fh:
        return pickle.load(fh)

def save_pickle(theobject, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(theobject, fh)

def build_base_pokemon_df(pokedex='galar'):
	# Pull Galar pokemon data from Pokemon API
    g = pb.pokedex(pokedex)
    ss_pokemons = []
    for entry in g.pokemon_entries:
        species = entry.pokemon_species
        print("{} {}".format(species.name, entry.entry_number))
        variety = None
    #     print("Species {} Number of varieties: {}".format(species.name,len(species.varieties)))
        for v in species.varieties:
            if "galar" in v.pokemon.name:
                variety = v
                break
        if variety is None:
            for v in species.varieties:
                if v.is_default:
                    variety = v
                    break

        if variety is None:
            print("No default variety for " + str(species))
            continue
        
        p = v.pokemon
        
        ss_entry = {'name':p.name,
                    'id':p.id,
                    'pokedex_number':entry.entry_number,
                    'generation':species.generation.id,
                    'is_legendary':species.is_legendary,
                    'is_mythical':species.is_legendary,
                    'species_id':species.id
                   }
        
        for s in p.stats:
            ss_entry[s.stat.name] = s.base_stat
            
        for t in p.types:
            ss_entry['type_{}'.format(t.slot)] = t.type.name

        ss_pokemons.append(ss_entry)

    ss_df = pd.DataFrame(ss_pokemons)
    ss_df = ss_df.rename(columns={'special-attack':'sp_attack','special-defense':'sp_defense'})

    return ss_df
    

def print_evolution_chain(c, indent_level=0):
    print("   "*indent_level + str(c.species.id) + " " + c.species.name)
    for t in c.evolves_to:
        print_evolution_chain(t, indent_level+1)


def get_pokemon_chain_info(chain, base_species_id=None, base_species_name=None):
    if base_species_id is None:
        base_species_id = chain.species.id
        base_species_name = chain.species.name
    
    if len(chain.evolves_to) == 0:
        is_final_form = True
    else:
        is_final_form = False
        
    chain_info = [{'species_id':chain.species.id, 
                   'name':chain.species.name, 
                   'base_species_id':base_species_id, 
                   'base_species_name':base_species_name,
                   'is_final_form':is_final_form}]
    for t in chain.evolves_to:
        chain_info.extend(get_pokemon_chain_info(t, base_species_id, base_species_name))
    
    return chain_info

def build_evolution_df():
    ECs = pb.APIResourceList('evolution-chain')
    chain_info =[]

    for i in list(ECs.names):
        logging.debug(i)
        print(i)
        # try:
        c = pb.evolution_chain(int(i)).chain
        # except:
            # continue
        chain_info.extend(get_pokemon_chain_info(c))
        
    chain_df = pd.DataFrame(chain_info)
    return chain_df

def print_species_varieties(pokedex):
    for entry in pokedex.pokemon_entries:
        species = entry.pokemon_species
        print("{} {}".format(entry.entry_number, species.name))
        variety = None
    #     print("Species {} Number of varieties: {}".format(species.name,len(species.varieties)))
        for v in species.varieties:
            print("  "+v.pokemon.name)

def draw_pokemon(id):
    a = pb.pokemon(id)
    display(Image(url=getattr(a.sprites.other,'official-artwork').front_default))

def add_zscores(df,stat_names):
    # Add z scores
    for s in stat_names:
        stdev = np.std(df.loc[:,s])
        mean = np.mean(df.loc[:,s])
        df.loc[:,"z_"+s] = (df.loc[:,s]-mean)/stdev    

    df.loc[:,'total_z'] = df[['z_' + n for n in stat_names]].sum(axis=1)

    return df

def is_pareto_efficient(costs, flex=0):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=(c+flex), axis=1)  # Remove dominated points
    return is_efficient


def verbose_poke_pareto(poke_df, flex=0):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = poke_df.loc[:,stat_names].values
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            for j, d in enumerate(costs):
                if is_efficient[j]:
                    is_efficient[is_efficient] = np.any(costs[is_efficient]<=(c+flex), axis=1)  # Remove dominated points
    return is_efficient

def vulnerability_str(pokemon_record, vul_cols):
    vulnerable = pokemon_record.loc[vul_cols]> 1
    vul_str = 'Vulnerable to ' + ', '.join([a[4:] for a in np.array(vul_cols)[vulnerable]])
    super_vulnerable = pokemon_record.loc[vul_cols]> 2
    if np.sum(super_vulnerable) > 0:
        supervul_str = '. Super vulnerable to ' + ', '.join([a[4:] for a in np.array(vul_cols)[super_vulnerable]])
    else:
        supervul_str = ''
    return vul_str + supervul_str

