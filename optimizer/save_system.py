"""
Universal save/load system for optimization methods.

Features:
- Saves progress every N iterations (configurable, default 10)
- Standardized format works across all optimizers (PSO, DE, hybrid, etc.)
- Tracks optimization history and metadata
- Efficient loading for different optimization methods
- Automatic backup rotation
"""

import os
import json
import time
import shutil
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class SaveState:
    """Standard save state format for all optimizers."""
    
    # Core optimization data
    problem_id: int
    optimizer_type: str  # "pso", "de", "hybrid", "block", etc.
    iteration: int
    best_fitness: float
    best_position: List[float]
    
    # History and statistics
    fitness_history: List[float]
    eval_count: int
    elapsed_time: float
    
    # Optimizer-specific state (positions, velocities, population, etc.)
    optimizer_state: Dict[str, Any]
    
    # Metadata
    save_timestamp: str
    total_iterations: int
    dimensions: int
    
    # Optional best strategy for final results
    best_strategy: Optional[Dict[str, Any]] = None
    best_times: Optional[List[float]] = None


class UniversalSaveSystem:
    """Universal save/load system for all optimization methods."""
    
    def __init__(self, 
                 problem_id: int,
                 models_dir: Optional[str] = None,
                 save_interval: int = 10,
                 max_backups: int = 5):
        """
        Args:
            problem_id: Problem identifier (2, 3, 4, 5)
            models_dir: Directory for saving models (default: ../models)
            save_interval: Save every N iterations
            max_backups: Maximum number of backup files to keep
        """
        self.problem_id = problem_id
        self.save_interval = save_interval
        self.max_backups = max_backups
        
        if models_dir is None:
            # Default to models/ directory relative to this file's parent
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Threading lock for concurrent saves
        self._save_lock = threading.Lock()
    
    def get_save_path(self, suffix: str = "") -> str:
        """Get the standardized save file path."""
        filename = f"problem{self.problem_id}_progress"
        if suffix:
            filename += f"_{suffix}"
        return os.path.join(self.models_dir, f"{filename}.json")
    
    def should_save(self, iteration: int) -> bool:
        """Check if this iteration should trigger a save."""
        return iteration > 0 and iteration % self.save_interval == 0
    
    def save_progress(self, save_state: SaveState, force: bool = False) -> bool:
        """
        Save optimization progress.
        
        Args:
            save_state: Current optimization state
            force: Force save even if not at save interval
            
        Returns:
            True if save was successful
        """
        if not force and not self.should_save(save_state.iteration):
            return False
            
        try:
            with self._save_lock:
                # Convert to dict for JSON serialization
                data = asdict(save_state)
                
                # Add schema version for future compatibility
                data["schema_version"] = "2.0"
                
                # Get save paths
                main_path = self.get_save_path()
                backup_path = self.get_save_path(f"backup_{int(time.time())}")
                
                # Create backup of existing file
                if os.path.exists(main_path):
                    shutil.copy2(main_path, backup_path)
                    self._cleanup_backups()
                
                # Save new state
                with open(main_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"[SaveSystem] Saved progress: iter {save_state.iteration}/{save_state.total_iterations}, "
                      f"fitness {save_state.best_fitness:.6f}")
                return True
                
        except Exception as e:
            print(f"[SaveSystem] Save failed: {e}")
            return False
    
    def load_progress(self, optimizer_type: Optional[str] = None) -> Optional[SaveState]:
        """
        Load the most recent optimization progress.
        
        Args:
            optimizer_type: If specified, prefer states from this optimizer type
            
        Returns:
            SaveState if found, None otherwise
        """
        try:
            main_path = self.get_save_path()
            
            if not os.path.exists(main_path):
                print(f"[SaveSystem] No save file found at {main_path}")
                return None
            
            with open(main_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to SaveState
            # Handle optional fields that might not exist
            save_state = SaveState(
                problem_id=data['problem_id'],
                optimizer_type=data['optimizer_type'],
                iteration=data['iteration'],
                best_fitness=data['best_fitness'],
                best_position=data['best_position'],
                fitness_history=data['fitness_history'],
                eval_count=data['eval_count'],
                elapsed_time=data['elapsed_time'],
                optimizer_state=data['optimizer_state'],
                save_timestamp=data['save_timestamp'],
                total_iterations=data['total_iterations'],
                dimensions=data['dimensions'],
                best_strategy=data.get('best_strategy'),
                best_times=data.get('best_times')
            )
            
            print(f"[SaveSystem] Loaded progress: {save_state.optimizer_type} iter {save_state.iteration}, "
                  f"fitness {save_state.best_fitness:.6f}")
            return save_state
            
        except Exception as e:
            print(f"[SaveSystem] Load failed: {e}")
            return None
    
    def _cleanup_backups(self):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            # Find all backup files for this problem
            backup_pattern = f"problem{self.problem_id}_progress_backup_"
            backup_files = []
            
            for filename in os.listdir(self.models_dir):
                if filename.startswith(backup_pattern) and filename.endswith('.json'):
                    filepath = os.path.join(self.models_dir, filename)
                    timestamp_str = filename[len(backup_pattern):-5]  # Remove .json
                    try:
                        timestamp = int(timestamp_str)
                        backup_files.append((timestamp, filepath))
                    except ValueError:
                        continue
            
            # Sort by timestamp (newest first) and remove old ones
            backup_files.sort(reverse=True)
            for _, filepath in backup_files[self.max_backups:]:
                os.remove(filepath)
                
        except Exception as e:
            print(f"[SaveSystem] Backup cleanup failed: {e}")
    
    def get_best_across_methods(self) -> Optional[Tuple[float, List[float], str]]:
        """
        Get the best result across all optimization methods for this problem.
        
        Returns:
            (best_fitness, best_position, method_name) or None
        """
        try:
            # Check both the progress file and legacy unified files
            best_fitness = None
            best_position = None
            best_method = None
            maximize = True  # Default assumption
            
            # Check progress file
            save_state = self.load_progress()
            if save_state:
                best_fitness = save_state.best_fitness
                best_position = save_state.best_position
                best_method = save_state.optimizer_type
            
            # Check legacy unified file
            unified_path = os.path.join(self.models_dir, f"problem{self.problem_id}_latest.json")
            if os.path.exists(unified_path):
                with open(unified_path, 'r', encoding='utf-8') as f:
                    unified_data = json.load(f)
                
                unified_fitness = unified_data.get('best_fitness')
                if unified_fitness is not None:
                    if best_fitness is None:
                        best_fitness = unified_fitness
                        best_position = unified_data.get('best_position', [])
                        best_method = "legacy"
                    else:
                        # Compare with current best
                        better = (unified_fitness > best_fitness) if maximize else (unified_fitness < best_fitness)
                        if better:
                            best_fitness = unified_fitness
                            best_position = unified_data.get('best_position', [])
                            best_method = "legacy"
            
            if best_fitness is not None:
                return best_fitness, best_position, best_method
            
            return None
            
        except Exception as e:
            print(f"[SaveSystem] Error getting best result: {e}")
            return None


class OptimizationTracker:
    """Helper class to integrate save system with optimization loops."""
    
    def __init__(self, save_system: UniversalSaveSystem, optimizer_type: str, total_iterations: int, dimensions: int):
        self.save_system = save_system
        self.optimizer_type = optimizer_type
        self.total_iterations = total_iterations
        self.dimensions = dimensions
        self.start_time = time.time()
        
        # Try to load previous progress
        self.loaded_state = save_system.load_progress(optimizer_type)
        self.starting_iteration = self.loaded_state.iteration if self.loaded_state else 0
    
    def should_resume(self) -> bool:
        """Check if we should resume from a previous state."""
        return self.loaded_state is not None and self.loaded_state.iteration < self.total_iterations
    
    def get_resume_data(self) -> Optional[Dict[str, Any]]:
        """Get data needed to resume optimization."""
        if self.loaded_state:
            return {
                'iteration': self.loaded_state.iteration,
                'best_fitness': self.loaded_state.best_fitness,
                'best_position': np.array(self.loaded_state.best_position),
                'fitness_history': self.loaded_state.fitness_history.copy(),
                'eval_count': self.loaded_state.eval_count,
                'optimizer_state': self.loaded_state.optimizer_state
            }
        return None
    
    def save_iteration(self, iteration: int, best_fitness: float, best_position: np.ndarray, 
                      fitness_history: List[float], eval_count: int, optimizer_state: Dict[str, Any],
                      best_strategy: Optional[Dict[str, Any]] = None, best_times: Optional[List[float]] = None,
                      force: bool = False) -> bool:
        """Save current optimization state."""
        
        save_state = SaveState(
            problem_id=self.save_system.problem_id,
            optimizer_type=self.optimizer_type,
            iteration=iteration,
            best_fitness=best_fitness,
            best_position=best_position.tolist(),
            fitness_history=fitness_history.copy(),
            eval_count=eval_count,
            elapsed_time=time.time() - self.start_time,
            optimizer_state=optimizer_state,
            save_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_iterations=self.total_iterations,
            dimensions=self.dimensions,
            best_strategy=best_strategy,
            best_times=best_times
        )
        
        return self.save_system.save_progress(save_state, force=force)


def create_save_system(problem_id: int, save_every: int = 10) -> UniversalSaveSystem:
    """Convenience function to create a save system."""
    return UniversalSaveSystem(problem_id=problem_id, save_interval=save_every)


def create_tracker(save_system: UniversalSaveSystem, optimizer_type: str, 
                  total_iterations: int, dimensions: int) -> OptimizationTracker:
    """Convenience function to create an optimization tracker."""
    return OptimizationTracker(save_system, optimizer_type, total_iterations, dimensions)