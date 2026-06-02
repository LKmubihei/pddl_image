(define (problem picture_74)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery blue_pump green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_pump)
        (clear blue_battery)
        (clear green_regulator)
        (part_at blue_pump table)
        (part_at blue_battery table)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at green_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
