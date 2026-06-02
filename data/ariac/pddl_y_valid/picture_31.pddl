(define (problem picture_31)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery green_regulator blue_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear green_regulator)
        (clear blue_pump)
        (part_at green_regulator table)
        (part_at green_battery table)
        (part_at blue_pump table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at green_regulator regulator_placement)
            (part_at green_battery battery_placement)
        )
    )
)
