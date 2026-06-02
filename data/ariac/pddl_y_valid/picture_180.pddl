(define (problem picture_180)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_battery blue_regulator blue_pump green_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_battery)
        (clear green_battery)
        (clear blue_regulator)
        (on blue_battery blue_pump)
        (part_at green_battery table)
        (part_at blue_regulator table)
        (part_at blue_pump table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at blue_regulator regulator_placement)
            (part_at blue_battery battery_placement)
        )
    )
)
