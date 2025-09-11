from datasets import load_dataset

def test_glue_loading():
    """Test loading GLUE datasets with the correct namespace"""
    
    tasks = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli']
    
    print("Testing GLUE dataset loading...")
    print("=" * 50)
    
    successful_tasks = []
    failed_tasks = []
    
    for task in tasks:
        print(f"\nTesting {task}...")
        
        try:
            # Load a small sample to test
            dataset = load_dataset('nyu-mll/glue', task, split='train[:5]')
            
            print(f"✅ {task} loaded successfully!")
            print(f"   Sample count: {len(dataset)}")
            
            if len(dataset) > 0:
                print(f"   Sample: {dataset[0]}")
            
            successful_tasks.append(task)
            
        except Exception as e:
            print(f"❌ {task} failed: {e}")
            failed_tasks.append(task)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successful tasks ({len(successful_tasks)}): {successful_tasks}")
    print(f"Failed tasks ({len(failed_tasks)}): {failed_tasks}")
    
    return successful_tasks, failed_tasks

if __name__ == "__main__":
    test_glue_loading()